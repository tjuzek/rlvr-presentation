## The Verifiability Landscape

### Where can RLVR help &mdash; and where can't it?

<div class="spectrum">
<div class="col green">

**Fully Verifiable**

- Arithmetic / grade-school math
- Competition mathematics
- Code generation (test suites)
- Format constraints (word count, JSON, case)

RLVR potential: **Proven**

</div>
<div class="col orange">

**Partially Verifiable**

- Factual QA (knowledge bases)
- Logical reasoning (proof checkers)
- Translation (reference-based metrics)
- Structured data extraction

RLVR potential: **Medium** &mdash; with caveats

</div>
<div class="col red">

**Fundamentally Subjective**

- Open-ended dialogue / chat
- Creative writing / style
- Ethical & moral reasoning
- Aesthetic judgment

RLVR potential: **Minimal** &mdash; need human judgment

</div>
</div>

### Theoretical limits

- RLVR works best where **cheap, reliable verifiers** exist
- As tasks get harder, the **verifier approaches the difficulty of the task itself**
  - Verifying a novel proof requires a proof checker as sophisticated as the prover
  - Verifying "good legal advice" requires a lawyer
- The boundary: **aesthetics, taste, ethics, and creativity** cannot be verified algorithmically

<aside class="notes">
This is the key conceptual slide. It frames the entire talk: RLVR is powerful but bounded. Everything to the right of the spectrum still needs humans.
</aside>
