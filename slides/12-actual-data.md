## The Actual Data Structure

This is what RLVR training data looks like on HuggingFace:

<div class="columns-3">
<div>

### GSM8K

`allenai/RLVR-GSM` &middot; 7,470 rows

```json
{
  "messages": [{
    "role": "user",
    "content": "There are 15 trees
      in the grove. Workers will
      plant trees today. After,
      there will be 21 trees. How
      many did they plant?"
  }],
  "ground_truth": "6",
  "dataset": "gsm8k"
}
```

4 fields. That's it.

</div>
<div>

### MATH

`allenai/RLVR-MATH` &middot; 7,500 rows

```json
{
  "messages": [{
    "role": "user",
    "content": "Find the domain of
      sqrt(x-2) / sqrt(5-x)"
  }],
  "ground_truth": "[2, 5)",
  "dataset": "MATH",
  "constraint_type": null,
  "constraint": null
}
```

Ground truth is a math expression.

</div>
<div>

### IFEval

`allenai/RLVR-IFeval` &middot; 14,973 rows

```json
{
  "messages": [{
    "role": "user",
    "content": "Describe IPv6.
      Your entire response should
      be in all lowercase letters."
  }],
  "ground_truth":
    "{\"func_name\":
      \"validate_lowercase\"}",
  "constraint_type":
    "All Lowercase"
}
```

Ground truth is a **function name**.

</div>
</div>

24 constraint types: lowercase, word count, paragraph count, JSON format, keyword frequency, forbidden words...

**Punchline**: ~30K JSON rows. Each row is a prompt + a way to check the answer. That's all RLVR needs.

<aside class="notes">
The mundanity is the point. This isn't exotic data -- it's problems with checkable answers. The power is in how you use it.
</aside>
