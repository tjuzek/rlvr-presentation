## The Verifier Landscape

### Verifiers come in many forms

| Domain | Verifier type | Example |
|--------|--------------|---------|
| **Math** (Tulu 3) | Exact-match | Extract last number, compare to gold |
| **Code** (DeepSeek-R1) | Test suite | Run unit tests, pass/fail |
| **Instruction following** (Tulu 3) | Constraint check | Count paragraphs, check case |

### The contrast with RLHF

| | RLHF | RLVR |
|---|------|------|
| **Reward signal** | Learnt neural network (~7B params) | 5-line Python function |
| **Training data** | Thousands of human preference pairs | Just the verification logic |
| **Cost** | $$$$ (human annotators) | ~$0 (code) |
| **Failure mode** | Reward hacking | Verifier gaming (but more detectable) |

<aside class="notes">
Code-based RLVR (DeepSeek-R1 at scale) uses test suites as verifiers — generate code, run tests, reward on pass/fail. Same RLVR principle, different domain.
</aside>
