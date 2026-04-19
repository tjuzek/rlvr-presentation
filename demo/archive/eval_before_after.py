"""
Quick evaluation script to compare a base model vs an RLVR-trained model
on GSM8K test problems. Use this if you've already trained a model
and just want to show the comparison.

Usage:
    python eval_before_after.py --base Qwen/Qwen2.5-1.5B-Instruct --trained ./rlvr-trained-model
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifiers import verify_gsm8k


def load_examples(n: int = 20):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        answer = item["answer"].split("####")[-1].strip()
        examples.append({"question": item["question"], "answer": answer})
    return examples


def evaluate_model(model_name: str, examples: list, label: str):
    print(f"\nLoading {label}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    correct = 0
    results = []
    t0 = time.time()

    for ex in examples:
        prompt = f"Solve this math problem step by step.\n\nQuestion: {ex['question']}\n\nShow your work and give the final numerical answer."
        messages = [{"role": "user", "content": prompt}]

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        reward = verify_gsm8k(response, ex["answer"])
        is_correct = reward > 0
        if is_correct:
            correct += 1
        results.append({"question": ex["question"][:60], "expected": ex["answer"], "response": response[:150], "correct": is_correct})

    elapsed = time.time() - t0
    acc = correct / len(examples) * 100
    print(f"  {label} accuracy: {acc:.1f}% ({correct}/{len(examples)}) in {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()
    return results, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--trained", default="./rlvr-trained-model")
    parser.add_argument("--n", type=int, default=20, help="Number of test examples")
    args = parser.parse_args()

    examples = load_examples(args.n)
    print(f"Evaluating on {len(examples)} GSM8K test problems\n")

    base_results, base_acc = evaluate_model(args.base, examples, "Base model")
    trained_results, trained_acc = evaluate_model(args.trained, examples, "RLVR-trained")

    print(f"\n{'='*70}")
    print(f"COMPARISON: Base {base_acc:.1f}% -> RLVR {trained_acc:.1f}% ({trained_acc - base_acc:+.1f}%)")
    print(f"{'='*70}")

    for i in range(min(5, len(examples))):
        b, a = base_results[i], trained_results[i]
        print(f"\nQ: {b['question']}... (answer: {b['expected']})")
        print(f"  Base  [{'OK' if b['correct'] else 'WRONG'}]: {b['response'][:120]}")
        print(f"  RLVR  [{'OK' if a['correct'] else 'WRONG'}]: {a['response'][:120]}")


if __name__ == "__main__":
    main()
