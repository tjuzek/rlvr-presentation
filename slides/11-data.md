## The RLVR Training Data

The entire dataset is remarkably small: **~30,000 prompts**.

| Dataset | Count | Domain | Verification |
|---------|------:|--------|-------------|
| **GSM8K** | 7,473 | Grade school math | Exact match on extracted number |
| **MATH** | 7,500 | Competition math | Exact match (flexible extraction) |
| **IF verifiable** | 14,973 | Instruction following | Programmatic constraint checkers |
| *Total* | *29,946* | | |

### Concrete examples

**GSM8K**: *"Natalia sold clips to 48 friends in April, then half as many in May. Total?"* &rarr; extract last number, check `== 72`

**IF constraint**: *"Write a poem about autumn. Exactly 4 paragraphs."* &rarr; count paragraphs, check `== 4`

**Key insight**: remarkably small dataset, remarkably simple verifiers.

<aside class="notes">
Compare to SFT which uses ~940K prompts, or DPO which uses ~350K preference pairs. RLVR uses just 30K prompts with verifiers. Full GSM8K problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" Full IF: "Write a poem about autumn. Your response must contain exactly 4 paragraphs."
</aside>
