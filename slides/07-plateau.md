## The Preference Tuning Plateau

RLHF and DPO made models broadly helpful &mdash; but not deeply expert.

<div class="plateau-grid">
  <div class="plateau-card">
    <div class="plateau-icon">&#x1F4CA;</div>
    <div class="plateau-label">General knowledge</div>
    <div class="plateau-score good">Above average</div>
  </div>
  <div class="plateau-card">
    <div class="plateau-icon">&#x1F4AC;</div>
    <div class="plateau-label">Conversation</div>
    <div class="plateau-score good">Above average</div>
  </div>
  <div class="plateau-card">
    <div class="plateau-icon">&#x1F9EE;</div>
    <div class="plateau-label">Math reasoning</div>
    <div class="plateau-score mid">Average</div>
  </div>
  <div class="plateau-card">
    <div class="plateau-icon">&#x1F4BB;</div>
    <div class="plateau-label">Code correctness</div>
    <div class="plateau-score mid">Average</div>
  </div>
  <div class="plateau-card">
    <div class="plateau-icon">&#x2696;&#xFE0F;</div>
    <div class="plateau-label">Formal reasoning</div>
    <div class="plateau-score mid">Average</div>
  </div>
</div>

**The problem:** preference-tuned models learnt to *sound* right, not to *be* right.

They became **(great) jacks of all trades, but masters of none.**

What if we could train on tasks where *correctness is measurable?*

<aside class="notes">
This is the narrative hinge. Preference tuning hit a ceiling on precision tasks because the reward signal is too noisy — a human annotator might prefer a well-written wrong answer over a terse correct one. RLVR breaks through this ceiling.
</aside>
