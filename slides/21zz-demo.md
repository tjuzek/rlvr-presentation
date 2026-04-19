## Live Demo: RLVR Training

**Setup**: OLMo-1B or Qwen2.5-0.5B &middot; 200 GSM8K problems &middot; exact-match verifier &middot; Lambda A10

### What we'll do

1. Evaluate the **base model** on 50 GSM8K test problems
2. Run RLVR training for a few hundred steps
3. Evaluate the **trained model** on the same problems
4. Compare outputs **side by side**

<div class="comparison">
<div>

#### Before RLVR

*"Natalia sold clips to 48 friends in April..."*

"Natalia is a good person who sells clips. She has many friends. The weather in April is nice for selling clips..."

</div>
<div>

#### After RLVR

*"Natalia sold clips to 48 friends in April..."*

"In April, Natalia sold 48 clips. In May, she sold 48/2 = 24 clips. In total: 48 + 24 = **72 clips**."

</div>
</div>

*Demo code in `demo/rlvr_demo.py`*

<aside class="notes">
Have the demo pre-kicked-off and show the training curves live. If live training isn't ready, show pre-recorded results and the code walkthrough.
</aside>
