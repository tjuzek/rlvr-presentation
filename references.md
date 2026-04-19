# Core References: RLVR Presentation

## The Primary Source
1. Lambert, Morrison, Pyatkin et al. **"Tulu 3: Pushing Frontiers in Open Language Model Post-Training."** arXiv:2411.15124, Nov 2024. [Paper](https://arxiv.org/abs/2411.15124) | [Blog](https://allenai.org/blog/tulu-3) | [Code](https://github.com/allenai/open-instruct) | [Data](https://huggingface.co/collections/allenai/tulu-3-datasets-673b8df14442393f7213f372)

## Foundations: RLHF & Post-Training
2. Ouyang et al. **"Training language models to follow instructions with human feedback."** NeurIPS 2022. [Paper](https://arxiv.org/abs/2203.02155) — The foundational RLHF paper (InstructGPT).
3. Christiano et al. **"Deep reinforcement learning from human preferences."** NeurIPS 2017. [Paper](https://arxiv.org/abs/1706.03741) — Original deep RL from human preferences.
4. Schulman et al. **"Proximal Policy Optimization Algorithms."** arXiv:1707.06347, 2017. [Paper](https://arxiv.org/abs/1707.06347) — PPO, the RL algorithm used in RLVR.
5. Ziegler et al. **"Fine-Tuning Language Models from Human Preferences."** arXiv:1909.08593, 2019. [Paper](https://arxiv.org/abs/1909.08593) — First PPO-based RLHF for language models.

## Preference Tuning
6. Rafailov et al. **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model."** NeurIPS 2024. [Paper](https://arxiv.org/abs/2305.18290) — DPO, the preference tuning method used before RLVR.
7. Stiennon et al. **"Learning to summarize with human feedback."** NeurIPS 2020. [Paper](https://arxiv.org/abs/2009.01325)

## Related Approaches: RL for Reasoning
8. Zelikman et al. **"STaR: Bootstrapping Reasoning with Reasoning."** NeurIPS 2022. [Paper](https://arxiv.org/abs/2203.14465) — Self-Taught Reasoner.
9. Zelikman et al. **"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking."** COLM 2024. [Paper](https://arxiv.org/abs/2403.09629)
10. Hoffman et al. **"Training chain-of-thought via latent-variable inference." (TRICE)** NeurIPS 2023. [Paper](https://openreview.net/forum?id=a147pIS2Co)
11. Kazemnejad et al. **"VinePPO: Unlocking RL Potential for LLM Reasoning through Refined Credit Assignment."** arXiv:2410.01679, 2024. [Paper](https://arxiv.org/abs/2410.01679)
12. Gehring et al. **"RLef: Grounding Code LLMs in Execution Feedback with Reinforcement Learning."** arXiv:2410.02089, 2024. [Paper](https://arxiv.org/abs/2410.02089)

## RLVR at Scale
13. DeepSeek-AI. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."** arXiv:2501.12948, Jan 2025. [Paper](https://arxiv.org/abs/2501.12948) — RLVR at massive scale with emergent reasoning.
14. Shao et al. **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models."** arXiv:2402.03300, 2024. [Paper](https://arxiv.org/abs/2402.03300) — Introduced GRPO.

## Key Benchmarks & Datasets
15. Cobbe et al. **"Training Verifiers to Solve Math Word Problems." (GSM8K)** arXiv:2110.14168, 2021. [Paper](https://arxiv.org/abs/2110.14168) | [Data](https://huggingface.co/datasets/openai/gsm8k)
16. Hendrycks et al. **"Measuring Mathematical Problem Solving with the MATH Dataset."** NeurIPS 2021. [Paper](https://arxiv.org/abs/2103.03874) | [Data](https://huggingface.co/datasets/hendrycks/competition_math)
17. Zhou et al. **"Instruction-Following Evaluation for Large Language Models." (IFEval)** arXiv:2311.07911, 2023. [Paper](https://arxiv.org/abs/2311.07911)

## Implementation & Infrastructure
18. Engstrom et al. **"Implementation Matters in Deep RL."** ICLR 2020. [Paper](https://openreview.net/forum?id=r1etN1rtPB) — Why PPO implementation details matter.
19. Huang et al. **"The N+ Implementation Details of RLHF with PPO."** COLM 2024. [Paper](https://openreview.net/forum?id=kHO2ZTa8e3) — Practical guide to PPO for LLMs.
20. Hu et al. **"OpenRLHF."** arXiv:2405.11143, 2024. [Paper](https://arxiv.org/abs/2405.11143) | [Code](https://github.com/OpenRLHF/OpenRLHF)
21. von Werra et al. **"TRL: Transformer Reinforcement Learning."** [Code](https://github.com/huggingface/trl)

## Open Models
22. Groeneveld et al. **"OLMo: Accelerating the Science of Language Models."** arXiv:2402.00838, 2024. [Paper](https://arxiv.org/abs/2402.00838) | [Models](https://huggingface.co/allenai)
23. Dubey et al. **"The Llama 3 Herd of Models."** arXiv:2407.21783, 2024. [Paper](https://arxiv.org/abs/2407.21783)

## Labour & Data Work
24. Perrigo, Billy. **"OpenAI Used Kenyan Workers on Less Than $2 Per Hour to Make ChatGPT Less Toxic."** *Time*, Jan 2023. [Article](https://time.com/6247678/openai-chatgpt-kenya-workers/) — Reporting on Sama-contracted Kenyan annotators and the human cost of RLHF-era data pipelines.
25. Dzieza, Josh. **"The Laid-off Scientists and Lawyers Training AI to Steal Their Careers."** *New York Magazine / Intelligencer* (in collaboration with *The Verge*), Mar 10 2026. [Article](https://nymag.com/intelligencer/article/white-collar-workers-training-ai.html) — On the white-collar expert gig economy (Mercor, Scale AI, Surge AI) producing RLVR-era training data. PDF: `articles/nymag-ai-educated-ghost-workers.pdf`.

## RLVR Training Datasets (Allen AI)
- [RLVR-GSM](https://huggingface.co/datasets/allenai/RLVR-GSM) — 7,470 train examples
- [RLVR-MATH](https://huggingface.co/datasets/allenai/RLVR-MATH) — 7,500 train examples
- [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) — 14,973 train examples
- [RLVR-GSM-MATH-IF-Mixed](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints) — Combined mix
