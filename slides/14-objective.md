## The RLVR Objective

### The math

$$\max_{\pi_\theta} \; \mathbb{E}_{y \sim \pi_\theta(x)} \left[ v(x, y) - \beta \, \text{KL}\left(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right) \right]$$

### Reading it term by term

*Reminder:* a **policy** $\pi$ is the LLM viewed as a probability distribution over completions &mdash; given a prompt $x$, it assigns a probability to every possible response $y$.

- $\max_{\pi_\theta}$ &mdash; we're optimising over the policy's weights $\theta$
- $\mathbb{E}_{y \sim \pi_\theta(x)}[\,\cdot\,]$ &mdash; the **expected value** of $[\,\cdot\,]$, averaged over completions $y$ sampled from the **current** policy (on-policy)
- $v(x, y)$ &mdash; the **verifiable reward**: $\alpha = 10$ if the verifier passes, $0$ otherwise
- $\pi_\theta(y|x)$ &mdash; probability the policy being trained assigns to completion $y$
- $\pi_{\text{ref}}(y|x)$ &mdash; same probability under the **frozen reference** (the SFT starting model)
- $\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ &mdash; how far the current policy has drifted from the reference
- $\beta$ &mdash; KL penalty coefficient ($\approx 0.05$ in Tulu 3) &mdash; the dial on the anchor

<aside class="notes">
Walk through the formula left to right. The 'max over theta' says we're adjusting weights. The expectation says we score the model's own samples — it's on-policy, the distribution changes as the model learns. v(x,y) is our verifier from the previous slides. The KL term compares token-level probabilities between the current policy and the frozen SFT checkpoint — zero if identical, larger as they diverge. Beta = 0.05 in Tulu 3 is empirically chosen: too small and the model collapses to reward hacks, too large and it can't learn. Next slide explains why the KL term is there in the first place.
</aside>
