## The Problem with RLHF

Traditional alignment requires **human preference data**:

1. Generate two responses to the same prompt
2. Ask a human: *"Which response is better?"*
3. Train a **reward model** on these comparisons, then optimise the LM against it using PPO

### What's wrong with this?

- **Expensive**: human labelling at scale costs millions
- **Slow**: feedback loops take weeks
- **Subjective**: different annotators disagree
- **Gameable**: models exploit the reward model without genuine improvement ("reward hacking")
- **Ethically fraught**: annotation is often outsourced to low-paid workers in the Global South under harsh conditions &mdash; *Time*'s reporting on OpenAI's Kenyan annotators at Sama (Perrigo, 2023) documented wages under US$2 per hour and sustained exposure to traumatic content.

<aside class="notes">
This sets up the motivation: RLHF is expensive, and the labour cost is borne disproportionately by workers in the Global South. Perrigo's Time piece is the canonical reference; Gray &amp; Suri ("Ghost Work") and Casilli provide the broader scholarly framing. Lan will cover a parallel but different labour story for the RLVR era later.
</aside>
