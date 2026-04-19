## The Last Mile Problem

Language models have improved at a striking pace &mdash; and the trajectory isn't slowing.

<div id="chart-error-rate" class="chart-container" style="height: 480px;"></div>

Flip the y-axis: plot **error rate** (100 &minus; score) on a log scale, for another perspective.

<aside class="notes">
Same data as the previous slide, but reframed. On a linear scale, 95 to 100 looks like a small gap. On a log scale of error rate, it's clear that each remaining percentage point is exponentially harder to close. The projection (roughly a straight line here, which means exponential decay of error) shows that even with consistent progress, we're asymptotically approaching zero — never reaching it. This is the "last mile problem" that motivates RLVR: we need training techniques that can squeeze out gains at the margin, where human preference data becomes too noisy to help.
</aside>
