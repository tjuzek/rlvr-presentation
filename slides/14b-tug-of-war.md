## Reward vs. KL: A Tug of War

| Force | Pulls toward | Without it |
|---|---|---|
| $+\,v(x,y)$ | Completions the verifier accepts | No learning signal |
| $-\,\beta\,\text{KL}$ | The starting model's distribution | Catastrophic forgetting, reward hacking |

### Tuning $\beta$

The two forces are in constant tension throughout training. Tulu 3 picks $\beta \approx 0.05$ &mdash; small enough that reward dominates on correct answers, large enough that wandering into off-distribution text gets punished before it pays off.

Too small $\Rightarrow$ model collapses to reward hacks.
Too large $\Rightarrow$ model refuses to explore and can't learn.

### Optimisation

Solved with **PPO** (Schulman et al. 2017) &mdash; the clipped surrogate keeps each gradient step small so the tug of war stays stable across thousands of updates. GRPO (DeepSeek, 2024) drops the value network and normalises rewards within each sampled group; same objective, lighter implementation.

<aside class="notes">
Beta is a hyperparameter — the Tulu 3 paper reports 0.05 after sweeping. Different papers pick differently: DeepSeek-R1 uses a KL schedule that decreases over training. The key insight is that PPO clipping and the KL penalty do DIFFERENT jobs: clipping bounds per-step policy change (stability), KL bounds total drift from the reference (prevents collapse). Both are needed. GRPO's innovation is replacing PPO's value network with within-group reward normalization — saves memory (no critic) without changing the objective itself. The formula we showed is faithful to both PPO and GRPO setups; the difference is purely in how the gradient is estimated.
</aside>
