"""
GRPO for RLVR — Pseudocode / Walkthrough

This file is NOT meant to be run. It's annotated pseudocode that shows
exactly how the GRPO algorithm applies to RLVR data, step by step.

For the real implementation, see:
  - train.py          (uses TRL's GRPOTrainer, which hides these internals)
  - verifier.py       (the reward function)

For the math, see conversation-notes.md.
"""

import torch
import torch.nn.functional as F

# ============================================================
# THE DATA — what we start with
# ============================================================

# This is one example from data/rlvr_math.json:
example = {
    "messages": [{"role": "user", "content": "Question: ... What is the remainder when x^2+7x-5 divides 2x^4+11x^3-42x^2-60x+47?"}],
    "ground_truth": "2x-8",       # <-- the verifier checks against this
    "dataset": "MATH",
    "constraint_type": None,
    "constraint": None,
}

# Note: there is NO reward in the data.
# The reward is computed on-the-fly during training.


# ============================================================
# THE VERIFIER — domain-specific correctness checker
# ============================================================

def verify_math(model_answer: str, ground_truth: str) -> float:
    """
    For MATH: extract the \\boxed{...} answer from the model's response,
    normalize it, and check if it matches ground_truth.

    Returns: 1.0 if correct, 0.0 if wrong.
    """
    extracted = extract_boxed_answer(model_answer)   # e.g., "2x-8"
    if math_equivalent(extracted, ground_truth):      # symbolic comparison
        return 1.0
    return 0.0

def verify_code(model_code: str, test_assertions: list[str]) -> float:
    """
    For code (MBPP): execute the generated code in a sandbox and run
    the test assertions against it.

    Returns: 1.0 if all tests pass, 0.0 if any fail.
    """
    try:
        exec_in_sandbox(model_code)
        for assertion in test_assertions:
            exec_in_sandbox(assertion)       # e.g., "assert add(1,2) == 3"
        return 1.0                           # all passed
    except:
        return 0.0                           # any failure


# ============================================================
# GRPO TRAINING LOOP
# ============================================================

def grpo_train(
    policy,              # the LLM we're training (pi_theta)
    ref_policy,          # frozen copy of the original SFT model (pi_ref)
    dataset,             # list of {messages, ground_truth, ...}
    G=8,                 # group size: completions per prompt
    epsilon=0.2,         # Proximal Policy Opt. clipping parameter
    beta=0.05,           # KL penalty coefficient
    lr=5e-6,             # learning rate
    num_steps=200,       # training steps
):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    for step in range(num_steps):

        # ------- Pick a batch of prompts -------
        batch = sample_batch(dataset, batch_size=4)

        all_losses = []

        for example in batch:
            prompt = example["messages"]
            ground_truth = example["ground_truth"]

            # =============================================
            # STEP 1: Generate a GROUP of completions
            # =============================================
            # The model tries to answer the same question G times.
            # We sample with temperature (stochastic) so we get variety.

            completions = []
            for _ in range(G):
                output = policy.generate(prompt, temperature=0.7)
                completions.append(output)

            # Example with G=4 for the math question above:
            #   completions[0]: "...therefore the remainder is \\boxed{2x-8}"     <- correct
            #   completions[1]: "...dividing, we get remainder \\boxed{2x + 3}"   <- wrong
            #   completions[2]: "...the answer is \\boxed{2x-8}"                  <- correct
            #   completions[3]: "...I think it's \\boxed{x-4}"                    <- wrong

            # =============================================
            # STEP 2: Score each completion with the verifier
            # =============================================
            # This is where the reward comes from — NOT from the data,
            # but from comparing the model's output to ground_truth.

            rewards = []
            for completion in completions:
                r = verify_math(completion, ground_truth)
                rewards.append(r)

            # rewards = [1.0, 0.0, 1.0, 0.0]
            #
            # This is the ENTIRE reward signal.
            # No human labeler. No reward model. Just: did you get it right?

            # =============================================
            # STEP 3: Compute group-relative advantages
            # =============================================
            # Instead of a critic network (like PPO), we use the group
            # itself as the baseline.

            mean_r = sum(rewards) / len(rewards)           # 0.5
            std_r = std(rewards)                            # 0.5
            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

            # advantages = [+1.0, -1.0, +1.0, -1.0]
            #
            # Correct answers get POSITIVE advantage  -> reinforce these
            # Wrong answers get NEGATIVE advantage     -> suppress these
            #
            # Key insight: if ALL answers were correct, mean=1, std=0,
            # advantages would be ~0 for everyone. Nothing to learn.
            # GRPO naturally focuses on problems where the model is uncertain.

            # =============================================
            # STEP 4: Compute the policy gradient loss
            # =============================================

            for i, completion in enumerate(completions):
                A_i = advantages[i]
                tokens = tokenize(completion)

                for t, token in enumerate(tokens):
                    # Probability ratio: how much has the policy changed
                    # since we generated these completions?
                    log_prob_new = policy.log_prob(token, context=prompt + tokens[:t])
                    log_prob_old = policy_old.log_prob(token, context=prompt + tokens[:t])
                    rho = torch.exp(log_prob_new - log_prob_old)

                    # Clipped surrogate objective (from PPO)
                    unclipped = rho * A_i
                    clipped = torch.clamp(rho, 1 - epsilon, 1 + epsilon) * A_i
                    policy_loss = -torch.min(unclipped, clipped)

                    # KL penalty: don't drift too far from the original model
                    log_prob_ref = ref_policy.log_prob(token, context=prompt + tokens[:t])
                    kl = log_prob_new - log_prob_ref    # per-token KL
                    kl_penalty = beta * kl

                    all_losses.append(policy_loss + kl_penalty)

        # =============================================
        # STEP 5: Update the model
        # =============================================
        total_loss = torch.mean(torch.stack(all_losses))
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Then: snapshot the current policy as policy_old for the next step


# ============================================================
# WHAT THIS LOOKS LIKE END-TO-END
# ============================================================
#
# Step 0: Start with an SFT model (e.g., Tulu 3 after supervised finetuning)
#         This model can already answer questions, just not very accurately.
#
# Step 1-200: For each training step:
#
#   DATA IN:    "What is the remainder when x^2+7x-5 divides 2x^4+..."
#               ground_truth = "2x-8"
#
#   MODEL:      Generates 8 attempts at the answer
#               Some correct (\\boxed{2x-8}), some wrong
#
#   VERIFIER:   Checks each: correct → reward 1.0, wrong → reward 0.0
#
#   GRPO:       Computes group-relative advantages
#               Updates policy to make correct reasoning more likely
#               KL penalty keeps the model from going off the rails
#
#   RESULT:     Model gradually gets better at math. On the Tulu 3 paper:
#               GSM8K went from 83.1% → 89.3% with RLVR
#               MATH went from 50% → 55% with RLVR
#
# ============================================================
# WHY THIS WORKS
# ============================================================
#
# 1. The model already "almost knows" the answer from SFT pretraining.
#    RLVR sharpens the distribution toward correct reasoning chains.
#
# 2. Group-relative advantage is self-calibrating:
#    - Easy problems (all correct): advantage ≈ 0, skip
#    - Hard problems (all wrong): advantage ≈ 0, skip
#    - Interesting problems (some right, some wrong): large signal, learn!
#
# 3. Binary reward is fine because the RL algorithm doesn't need
#    fine-grained scores — it just needs to know which outputs to
#    reinforce and which to suppress.
