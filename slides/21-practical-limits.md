## Practical Limitations

### Things that go wrong

**Overoptimisation**: the model games the verifier &mdash; reward goes up, but overall quality drops (Figure 21 in paper)

**Brittle verifiers**: exact-match rejects correct answers with different formatting (`42` vs `42.0` vs `\frac{42}{1}`)

**Saturation at scale**: at 70B, gains are already modest. Does RLVR keep helping as models get stronger?

**Compute cost**: PPO requires 4 models in memory (policy, reference, value, reward). At 70B: 48 GPUs, 60 hours

**Binary reward is information-poor**: `71` on a problem with answer `72` gets the same $0$ as `banana`

<aside class="notes">
These are real engineering challenges. The brittle verifier problem is particularly interesting -- MATH needed a "flex" extraction strategy with 3 different answer parsers. On overoptimisation: the paper shows this explicitly in Figure 21. On binary rewards: there is no gradient signal for partial correctness -- a close wrong answer and total nonsense are treated identically.
</aside>
