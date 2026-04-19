## Results: What RLVR Achieves

### Tulu 3 8B (Llama 3.1 &rarr; DPO &rarr; RLVR)

| Benchmark | Llama 3.1 | Tulu 3 DPO | **Tulu 3 RLVR** | &Delta; |
|-----------|----------:|----------:|-----------:|-----:|
| GSM8K | 83.4 | 84.3 | **87.6** | +3.3 |
| MATH | 42.5 | 42.0 | **43.7** | +1.7 |
| IFEval | 80.6 | 81.1 | **82.4** | +1.3 |
| BigBenchHard | 62.8 | 65.8 | **66.0** | +0.2 |
| DROP | 61.5 | 62.5 | **62.6** | +0.1 |
| AlpacaEval 2 | 24.2 | 33.5 | **34.5** | +1.0 |
| Safety | 75.2 | **87.2** | 85.5 | &mdash; |

### The surprise

RLVR was only trained on **math** and **instruction following** &mdash; yet it also improved **BigBenchHard**, **DROP**, and **AlpacaEval**. Hypothesis: verifiable tasks teach a general *reasoning discipline* that transfers. At 70B, gains are modest but positive.

<aside class="notes">
The generalisation to unseen tasks is perhaps the most interesting finding. It suggests RLVR doesn't just teach narrow skills but improves the model's general reasoning.
</aside>
