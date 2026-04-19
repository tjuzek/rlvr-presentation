## Why the KL Penalty?

$$\max_{\pi_\theta} \; \mathbb{E}_{y \sim \pi_\theta(x)} \left[ v(x, y) - \beta \, \text{KL}\left(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right) \right]$$

### Without it, the model *collapses*

It finds **degenerate shortcuts** to the reward: always outputting the same number, emitting a memorised correct-looking string, or exploiting a verifier weakness it discovered mid-training. Reward goes up; general capability craters.

### What the anchor does

The KL term says: *"be correct, but don't forget everything else you know."* It tethers the policy to the SFT checkpoint &mdash; the model that already speaks fluent English, follows instructions, and knows facts about the world. Drifting far from $\pi_{\text{ref}}$ is only worth it if the reward gain justifies the tether tension.

<aside class="notes">
Concretely what 'collapse' looks like: early RLVR experiments without KL saw models output '42' for every math problem (high mean reward on a dataset where 42 appears often), or the model would learn to produce a format that confused the extractor into reading a correct-looking number. The KL term prevents this by making weird output distributions expensive — even on a correct answer, if the policy puts 0.99 probability on tokens the reference model gave 0.01, the KL contribution eats into the reward. The model has to find completions that are BOTH high-reward AND not too surprising under the reference. Same mechanism RLHF uses; RLVR inherits it unchanged. Next slide: the tug-of-war visual and how beta is tuned.
</aside>
