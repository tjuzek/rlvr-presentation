## What Comes After RLVR?

RLVR works because verification is cheap. The field is now asking: **what other axes can we scale?**

| Direction | Key Idea | Who's Pushing It |
|-----------|----------|-----------------|
| **Inference-time compute** | Let models "think longer" &mdash; a second scaling axis independent of model size | OpenAI (o3/o4), DeepSeek-R1 |
| **Self-play &amp; autocurricula** | Models generate their own tasks, solve, and verify &mdash; zero human data | Absolute Zero (NeurIPS '25 spotlight) |
| **Process reward models** | Score each reasoning *step*, not just the final answer &mdash; denser signal | OpenAI, ICLR '25 best paper |
| **Agentic RL** | Train in multi-turn, tool-using environments &mdash; optimise for task completion | Microsoft Research, Agent-R1 |
| **Diffusion LLMs** | Generate all tokens in parallel, refine iteratively &mdash; 10&times; faster | Inception Labs (Mercury), Google |

### The meta-narrative

> "Pre-training as we know it will unquestionably end &hellip; we have but one internet."
> &mdash; Ilya Sutskever, NeurIPS 2024

The dominant paradigm is shifting from **scaling data** to **scaling algorithms**: smarter training signals, richer verification, and inference-time search. RLVR is the first step on that road.

<aside class="notes">
These are not speculative -- all five directions have published results or deployed systems as of early 2025. Inference-time compute is already shipping in o3/o4 and DeepSeek-R1. Self-play (Absolute Zero Reasoner) achieved SOTA with literally zero external training data. Process reward models give step-level credit assignment for long reasoning chains. Agentic RL extends RLVR from single-turn QA to multi-step tool use. Diffusion LLMs (Mercury) represent an architectural alternative to autoregressive generation. The Sutskever quote is from his NeurIPS 2024 keynote. Dario Amodei has also noted there's "no publicly known scaling law for RL" -- the RL frontier is still largely unexplored.
</aside>
