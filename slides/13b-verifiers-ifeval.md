## Verifiers in the Wild &mdash; IFEval

### A registry of ~25 parameterised constraint validators

```python
class KeywordFrequency(Instruction):
    def __init__(self, keyword: str, frequency: int, relation: str):
        self.keyword, self.frequency, self.relation = keyword, frequency, relation

    def check_following(self, value: str) -> bool:
        n = len(re.findall(rf"\b{re.escape(self.keyword)}\b", value, re.I))
        return n == self.frequency if self.relation == "exactly" else n >= self.frequency

INSTRUCTION_DICT = {
    "keywords:frequency":                   KeywordFrequency,
    "change_case:english_lowercase":        LowercaseInstruction,
    "length_constraints:number_paragraphs": NumberParagraphs,
    # … ~25 more
}

def verify_ifeval(response: str, row: dict) -> float:
    instruction = INSTRUCTION_DICT[row["func_name"]](**row["params"])
    return 10.0 if instruction.check_following(response) else 0.0
```

Each validator is a **class**, not a function &mdash; because constraints carry parameters (`keyword`, `frequency`, `relation`). The dispatcher reads `func_name` from the row's `ground_truth` blob, instantiates the right class with the right parameters, and calls `check_following`. Same pattern scales to code verification (add timeout + subprocess sandboxing) and to any new domain you can write a checker for.

<aside class="notes">
This is why IFEval validators are classes rather than plain functions — a constraint like "include 'sustainable' exactly 3 times" needs to carry the keyword and the frequency as state. The parameters live in the row's ground_truth JSON (we saw this earlier: '{"func_name": "validate_lowercase", ...}'). So the full data pipeline is: row contains constraint_type + func_name + params, trainer reads them, dispatcher instantiates the right Instruction subclass, calls check_following on the model output, returns 10.0 or 0.0. Same architecture applies to code verifiers — you just add subprocess sandboxing and a timeout (see our code-rlvr/verifier.py).
</aside>
