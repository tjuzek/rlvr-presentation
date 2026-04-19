## What Preference Data Looks Like

Every RLHF training example is a human judgment: *"Response A is better than Response B."*

<div class="columns-2">
<div class="pref-chosen">

#### Chosen (preferred)

**Prompt:** *"Explain photosynthesis in one sentence."*

> Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.

</div>
<div class="pref-rejected">

#### Rejected

**Prompt:** *"Explain photosynthesis in one sentence."*

> Plants use the sun to make food and also they release oxygen and stuff which is important for us to breathe.

</div>
</div>

#### Reward model training

- Collect **thousands** of these comparisons
- Train a neural network to better predict human preferences
- Use this learnt reward signal for RL

The reward model is expensive to build, subjective by nature, and gameable.

<aside class="notes">
This slide makes RLHF concrete before we introduce RLVR. The audience should feel the weight of the process: every training signal requires a human to read two responses and make a judgment call.
</aside>
