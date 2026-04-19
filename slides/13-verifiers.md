## The Verifier Functions

The verifiers are trivially simple Python functions:

### GSM8K Verifier

```python
import re

def verify_gsm8k(response: str, gold_answer: float) -> float:
    """Extract the last number from the response and check it."""
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    if not numbers:
        return 0.0  # No number found
    predicted = float(numbers[-1].replace(',', ''))
    return 10.0 if predicted == gold_answer else 0.0
```

### IF Constraint Verifier (example: lowercase)

```python
def validate_lowercase(response: str) -> float:
    """Check if the entire response is lowercase."""
    return 10.0 if response == response.lower() else 0.0
```

<aside class="notes">
The verifier functions are intentionally simple. The sophistication is in the RL training loop, not the reward signal. Synthetic data generation techniques (e.g., creating bug/fix pairs) are a separate, complementary approach that can feed into RLVR pipelines but are not RLVR themselves.
</aside>
