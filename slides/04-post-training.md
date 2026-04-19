## The Post-Training Pipeline

How do we get from a base model to a useful assistant?

<div class="pipeline">
  <div class="step">Pre-training<br><small>Learn language</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">SFT<br><small>Follow instructions</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Preference Tuning<br><small>Human preferences</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step active">RLVR<br><small>Get the right answer</small></div>
</div>

- **Pre-training**: learn language from trillions of tokens (internet text)
- **SFT** (Supervised Fine-Tuning): learn to follow instructions from curated examples
- **Preference Tuning** (DPO/RLHF): learn which answers humans *prefer*
- **RLVR**: learn to produce *correct* answers on verifiable tasks

The key ingredient of recent progress: **preference learning (PL)**.

- **Human PL**: annotators compare responses, train a reward model (RLHF/DPO)
- **Synthetic PL**: use model-generated preferences &mdash; cheaper but risky (self-consuming loops can degrade quality over generations)

Both approaches improved general helpfulness. But as we'll see, they hit a ceiling on precision tasks.

<aside class="notes">
This is the Tulu 3 recipe. Other labs follow similar pipelines. The key innovation is that final RLVR stage. Note the distinction between human and synthetic preference learning — synthetic PL is tempting because it's cheap, but self-consuming loops (model trains on its own outputs) can cause quality collapse. This is a known problem in the literature.
</aside>
