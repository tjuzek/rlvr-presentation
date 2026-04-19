## Training Dynamics

### What happens during RLVR training?

<div id="chart-rewards" class="chart-container" style="height:280px"></div>

Monitor three signals: **verifiable rewards** (should rise), **KL divergence** (needs control), and **response length** (can inflate as the model hedges).

### The $\beta$ trade-off

| $\beta$ (KL penalty) | Reward | KL Divergence | Overall Quality |
|----|--------|---------------|----------------|
| 0.01 (low) | Highest | Highest | **Worst** &mdash; overoptimisation |
| 0.03 | High | Moderate | Good |
| **0.05** | **Good** | **Controlled** | **Best** |
| 0.1 (high) | Lowest | Lowest | Moderate &mdash; too constrained |

**Lower $\beta$** allows higher reward but risks overoptimisation; **higher $\beta$** stays close but barely learns. Sweet spot: enough freedom to learn, not enough to forget.

<aside class="notes">
The charts.js file will render an interactive Plotly chart here showing reward curves for different beta values. The data comes from Figure 19 of the paper.
</aside>
