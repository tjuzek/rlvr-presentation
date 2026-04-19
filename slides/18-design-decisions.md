## Key Design Decisions

### Where to start RLVR training?

| Starting Point | KL for same reward | Final quality |
|---------------|-------------------|---------------|
| Base model | Huge | Poor |
| SFT checkpoint | Large | Moderate |
| **DPO checkpoint** | **Moderate** | **Best** |

Starting from a stronger model (DPO) gives a better launch point &mdash; the model already "knows how" to reason; RLVR sharpens it.

### Pure verifiable rewards > mixed signals

Adding trained reward model scores introduces **noise** and does **not** improve results. **Simpler is genuinely better.**

### Overoptimisation is real

As KL divergence grows, the target task can **improve** while average quality **drops**. Monitor **multiple metrics**, not just the target.

<aside class="notes">
The finding that pure binary rewards beat mixed rewards is important. It suggests that for verifiable tasks, the clean signal from ground truth is more useful than the noisy signal from a reward model.
</aside>
