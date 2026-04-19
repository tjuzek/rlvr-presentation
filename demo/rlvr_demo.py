"""
RLVR Demo: Train a small model with verifiable rewards on GSM8K.

This is a proof-of-concept demonstration of Reinforcement Learning
from Verifiable Rewards. It trains a small language model to improve
at grade-school math by giving it reward only when it gets the right answer.

Usage:
    # Install dependencies first:
    pip install torch transformers datasets trl accelerate

    # Run the demo:
    python rlvr_demo.py

    # Or run just the evaluation comparison:
    python rlvr_demo.py --eval-only

Hardware: Requires a GPU with at least 16GB VRAM for 1B models,
          or 40GB+ for 8B models.
"""

import argparse
import json
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifiers import verify_gsm8k


# ---- Configuration ----

# Small models for quick demo (pick one):
# MODEL_NAME = "allenai/OLMo-1B"          # Allen AI, fully open
# MODEL_NAME = "Qwen/Qwen2.5-0.5B"       # Smallest viable model
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Good balance of size/capability

NUM_TRAIN_EXAMPLES = 200    # Subset of GSM8K train
NUM_EVAL_EXAMPLES = 50      # Subset of GSM8K test
MAX_NEW_TOKENS = 256        # Max tokens per response
BATCH_SIZE = 4
NUM_TRAIN_STEPS = 200       # PPO steps (increase for better results)
LEARNING_RATE = 1e-6
KL_PENALTY_BETA = 0.05     # KL divergence coefficient
REWARD_VALUE = 10.0         # Reward for correct answers


def load_gsm8k_data(split: str, n: int):
    """Load n examples from GSM8K."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        # Extract the numerical answer from the solution
        answer = item["answer"].split("####")[-1].strip()
        examples.append({
            "question": item["question"],
            "answer": answer,
            "full_solution": item["answer"],
        })
    return examples


def format_prompt(question: str) -> str:
    """Format a GSM8K question as a chat prompt."""
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {question}\n\n"
        f"Show your work and give the final numerical answer."
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate(model, tokenizer, examples: list[dict]) -> dict:
    """Evaluate the model on a set of GSM8K examples."""
    correct = 0
    results = []

    for ex in examples:
        prompt = format_prompt(ex["question"])
        response = generate_response(model, tokenizer, prompt, MAX_NEW_TOKENS)
        reward = verify_gsm8k(response, ex["answer"], REWARD_VALUE)
        is_correct = reward > 0

        results.append({
            "question": ex["question"][:80] + "...",
            "expected": ex["answer"],
            "response": response[:200] + ("..." if len(response) > 200 else ""),
            "correct": is_correct,
        })

        if is_correct:
            correct += 1

    accuracy = correct / len(examples) * 100
    return {"accuracy": accuracy, "correct": correct, "total": len(examples), "results": results}


def print_comparison(before_results: list[dict], after_results: list[dict], n: int = 5):
    """Print side-by-side comparison of before/after responses."""
    print("\n" + "=" * 80)
    print("BEFORE vs AFTER RLVR TRAINING")
    print("=" * 80)

    for i in range(min(n, len(before_results))):
        b = before_results[i]
        a = after_results[i]
        status_b = "CORRECT" if b["correct"] else "WRONG"
        status_a = "CORRECT" if a["correct"] else "WRONG"

        print(f"\n--- Example {i+1} (expected: {b['expected']}) ---")
        print(f"BEFORE [{status_b}]: {b['response'][:150]}")
        print(f"AFTER  [{status_a}]: {a['response'][:150]}")


def main():
    parser = argparse.ArgumentParser(description="RLVR Demo on GSM8K")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, no training")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--train-steps", type=int, default=NUM_TRAIN_STEPS)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load data
    print("Loading GSM8K data...")
    train_data = load_gsm8k_data("train", NUM_TRAIN_EXAMPLES)
    eval_data = load_gsm8k_data("test", NUM_EVAL_EXAMPLES)
    print(f"  Train: {len(train_data)} examples, Eval: {len(eval_data)} examples")

    # Baseline evaluation
    print("\n--- Baseline Evaluation ---")
    t0 = time.time()
    baseline = evaluate(model, tokenizer, eval_data)
    print(f"Baseline accuracy: {baseline['accuracy']:.1f}% ({baseline['correct']}/{baseline['total']})")
    print(f"  (took {time.time()-t0:.1f}s)")

    if args.eval_only:
        # Show some example outputs
        for r in baseline["results"][:5]:
            status = "CORRECT" if r["correct"] else "WRONG"
            print(f"\n[{status}] Expected: {r['expected']}")
            print(f"  Response: {r['response'][:200]}")
        return

    # RLVR Training with TRL's PPO
    print("\n--- Starting RLVR Training ---")
    print(f"  Steps: {args.train_steps}, LR: {LEARNING_RATE}, Beta: {KL_PENALTY_BETA}")

    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

        # Wrap model with value head for PPO
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        ppo_config = PPOConfig(
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            mini_batch_size=BATCH_SIZE,
            ppo_epochs=1,
            init_kl_coef=KL_PENALTY_BETA,
            target_kl=None,
            log_with=None,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=ppo_model,
            tokenizer=tokenizer,
        )

        # Training loop
        reward_history = []
        for step in range(args.train_steps):
            # Sample a batch
            batch_indices = torch.randint(0, len(train_data), (BATCH_SIZE,)).tolist()
            batch = [train_data[i] for i in batch_indices]

            # Prepare queries
            queries = []
            for ex in batch:
                prompt = format_prompt(ex["question"])
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                query_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
                queries.append(query_ids)

            # Generate responses
            responses = []
            for q in queries:
                output = ppo_trainer.generate(q.unsqueeze(0).to(ppo_model.pretrained_model.device), max_new_tokens=MAX_NEW_TOKENS)
                response_ids = output[0][len(q):]
                responses.append(response_ids)

            # Compute verifiable rewards
            rewards = []
            for i, ex in enumerate(batch):
                response_text = tokenizer.decode(responses[i], skip_special_tokens=True)
                reward = verify_gsm8k(response_text, ex["answer"], REWARD_VALUE)
                rewards.append(torch.tensor(reward))

            # PPO step
            stats = ppo_trainer.step(queries, responses, rewards)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)
            reward_history.append(avg_reward)

            if (step + 1) % 10 == 0:
                recent_avg = sum(reward_history[-10:]) / min(10, len(reward_history))
                print(f"  Step {step+1}/{args.train_steps} | Avg reward: {recent_avg:.2f} | KL: {stats.get('objective/kl', 0):.3f}")

        # Save the trained model
        print("\nSaving trained model...")
        ppo_model.save_pretrained("./rlvr-trained-model")
        tokenizer.save_pretrained("./rlvr-trained-model")

        # Post-training evaluation
        print("\n--- Post-RLVR Evaluation ---")
        # Load the trained model for eval
        trained_model = ppo_model.pretrained_model
        after = evaluate(trained_model, tokenizer, eval_data)
        print(f"After RLVR accuracy: {after['accuracy']:.1f}% ({after['correct']}/{after['total']})")
        print(f"Improvement: {after['accuracy'] - baseline['accuracy']:+.1f}%")

        # Side-by-side comparison
        print_comparison(baseline["results"], after["results"])

        # Save reward history
        with open("reward_history.json", "w") as f:
            json.dump(reward_history, f)
        print("\nReward history saved to reward_history.json")

    except ImportError:
        print("\nTRL not installed. Install with: pip install trl")
        print("Showing baseline evaluation only.")
        for r in baseline["results"][:5]:
            status = "CORRECT" if r["correct"] else "WRONG"
            print(f"\n[{status}] Expected: {r['expected']}")
            print(f"  Response: {r['response'][:200]}")


if __name__ == "__main__":
    main()
