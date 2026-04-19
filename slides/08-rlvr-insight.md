## The RLVR Insight

Some tasks have **objectively verifiable answers**:

| Task | Example | Verification |
|------|---------|-------------|
| **Math** | "What is 234 &times; 17?" &rarr; 3,978 | Check the number |
| **Code** | "Write a sort function" | Run the test suite |
| **Instruction following** | "Write exactly 3 paragraphs" | Count paragraphs |

### The idea

Replace the **expensive reward model** with a **cheap verification function**:

<div class="reward-box">
  <div>Verifiable Reward</div>
  <div class="formula">r = 10 if correct, 0 otherwise</div>
</div>

No reward model, no human preferences &mdash; just **ground truth** and a function to check it.

This is **RLVR**: Reinforcement Learning from Verifiable Rewards.

<aside class="notes">
The beauty is in the simplicity. For any domain where we can check correctness programmatically, we can skip the entire RLHF apparatus.
</aside>
