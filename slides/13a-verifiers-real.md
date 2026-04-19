## Verifiers in the Wild &mdash; GSM8K

The previous slide is pedagogical &mdash; here's what production verifiers in [`allenai/open-instruct`](https://github.com/allenai/open-instruct) actually look like.

### Multi-stage answer extraction

```python
def extract_gsm_answer(text: str) -> str | None:
    # 1) '#### <answer>'  — GSM8K's native delimiter
    if m := re.search(r"####\s*(-?[\d,]+\.?\d*)", text):
        return m.group(1).replace(",", "")
    # 2) \boxed{…}          — reasoning-model convention
    if m := re.search(r"\\boxed\{([^}]+)\}", text):
        return m.group(1).strip().replace(",", "")
    # 3) Last number anywhere in the response
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None

def verify_gsm8k(response: str, gold: str) -> float:
    pred = extract_gsm_answer(response)
    try:
        return 10.0 if pred and float(pred) == float(gold) else 0.0
    except ValueError:
        return 0.0
```

Three fallbacks in order &mdash; because models format answers three different ways. Miss any one and a correct answer scores zero. MATH goes further still, using `math_verify`/`sympy` for symbolic equality.

<aside class="notes">
Backup slide for Q&A. If someone asks "is this the real code?": yes, this is representative of what's in allenai/open-instruct. The real GSM8K verifier tries three extraction strategies in order because models format answers differently — '#### 72' is GSM8K's native format, '\boxed{72}' is what reasoning-trained models produce, and the last-number fallback catches the rest. MATH uses sympy-based symbolic equality via math_verify (not regex) so that '1/2' and '0.5' and '\frac{1}{2}' all compare equal. The reward scale (10.0) matches Tulu 3's alpha=10. Next slide: the IFEval validator registry.
</aside>
