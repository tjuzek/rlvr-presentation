/**
 * Diffusion Text Transition Engine for RLVR Presentation
 *
 * Creates a "diffusion" effect where text resolves from block characters
 * (█ ▓ ░) on slide entrance, and encodes back to blocks on exit.
 * Mimics the noise-to-signal process of diffusion models.
 *
 * Enhanced with a "morphing bridge" transition: after encoding to blocks,
 * the character count smoothly morphs to a shared count K before the slide
 * swap, then morphs from K to the new slide's count before decoding. This
 * eliminates the visual jump when consecutive slides have different amounts
 * of text content.
 *
 * Charts use CSS-driven blur + fade + scale transitions (GPU-accelerated).
 *
 * Flow:  N chars → N blocks → K blocks → [swap] → K blocks → M blocks → M chars
 */

const BLOCK_CHARS = ['\u2588', '\u2593', '\u2591']; // █ ▓ ░

const DIFFUSION_CONFIG = {
    enterDuration: 300,      // decode from blocks to text
    exitDuration: 300,       // encode from text to blocks
    morphDuration: 140,      // morph block count N→K or K→M
    stepMs: 27,              // ms between block stages per character
    jitterRate: 0.06,        // probability of random block swap per frame while waiting
    mathFadeDelay: 280,      // ms before KaTeX fades in
    chartFadeDelay: 200,     // ms before charts fade in
    chartExitDuration: 400,  // ms for chart CSS exit transition (matches CSS)
    mediaFadeDelay: 150,     // ms before images fade in
    initialDelay: 80,        // ms delay before first slide animation starts
    morphStaggerMs: 12,      // ms stagger between block additions/removals in morph
};

class DiffusionEngine {
    constructor(reveal) {
        this._reveal = reveal;
        this._isAnimating = false;
        this._bypassNext = false;
        this._pendingNav = null;
        this._currentCleanup = null;
        // Cache of character counts per slide for pre-computation
        this._slideCharCounts = new Map();
        // Saved DOM snapshots: slide → innerHTML, taken before _wrapTextNodes
        // mutates the DOM.  _restoreDOM replays the snapshot so the slide is
        // returned to its exact pre-animation state regardless of what the
        // morph / encode / decode phases did to the tree.
        this._domSnapshots = new Map();
        // Animation frame handle for slide number effect
        this._slideNumberAnimFrame = null;
    }

    init() {
        this._reveal.on('ready', (event) => {
            // Pre-scan all slides to cache character counts
            this._prescanSlides();

            setTimeout(() => {
                const slide = this._reveal.getCurrentSlide();
                if (slide) this._runEntrance(slide, null);
            }, DIFFUSION_CONFIG.initialDelay);
        });

        this._reveal.on('beforeslidechange', (event) => {
            this._onBeforeSlideChange(event);
        });

        this._reveal.on('slidechanged', (event) => {
            this._onSlideChanged(event);
        });
    }

    // ─── Slide Pre-scanning ──────────────────────────────────────

    /**
     * Walk all slides and cache their non-whitespace character counts.
     * This lets us compute K (shared block count) before the animation starts,
     * without needing to peek at the next slide's DOM during the transition.
     */
    _prescanSlides() {
        const slides = this._reveal.getSlides();
        for (const slide of slides) {
            const count = this._countTextChars(slide);
            this._slideCharCounts.set(slide, count);
        }
    }

    /**
     * Count non-whitespace text characters in a slide, excluding skipped
     * elements (KaTeX, charts, images, notes, etc.). This mirrors the logic
     * of _collectTextNodes + _wrapTextNodes but without mutating the DOM.
     */
    _countTextChars(slide) {
        let count = 0;
        const walker = document.createTreeWalker(
            slide,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: (node) => {
                    if (this._shouldSkip(node.parentElement)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    if (!node.textContent.trim()) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );
        let node;
        while ((node = walker.nextNode())) {
            for (const ch of node.textContent) {
                if (ch !== ' ' && ch !== '\n' && ch !== '\t' && ch !== '\r') {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Compute K — the shared block count that bridges two slides.
     *
     * Uses the geometric mean of N and M, which handles asymmetric cases well
     * (e.g. title slide with 50 chars → dense slide with 600 chars: K ≈ 173,
     * which is a reasonable visual middle ground). Clamped to [min(N,M)*0.6,
     * max(N,M)*1.0] to avoid extremes.
     *
     * If either count is zero, returns the other (no morphing needed).
     */
    _computeSharedCount(n, m) {
        if (n === 0) return m;
        if (m === 0) return n;

        const geomean = Math.round(Math.sqrt(n * m));
        const lo = Math.round(Math.min(n, m) * 0.6);
        const hi = Math.max(n, m);
        return Math.max(lo, Math.min(hi, geomean));
    }

    // ─── Navigation Interception ─────────────────────────────────

    _onBeforeSlideChange(event) {
        // Let programmatic navigation through after exit animation
        if (this._bypassNext) {
            this._bypassNext = false;
            return;
        }

        // Block navigation during animation
        if (this._isAnimating) {
            event.preventDefault();
            return;
        }

        // Skip animation in overview mode
        if (this._reveal.isOverview()) {
            return;
        }

        // Don't animate for fragment navigation (same slide)
        const current = this._reveal.getIndices();
        if (event.indexh === current.h && event.indexv === current.v) {
            return;
        }

        // Intercept: run exit animation with morph bridge, then navigate
        event.preventDefault();

        const targetH = event.indexh;
        const targetV = event.indexv;
        const currentSlide = this._reveal.getCurrentSlide();

        if (!currentSlide) {
            this._bypassNext = true;
            this._reveal.slide(targetH, targetV);
            return;
        }

        // Look up the target slide to get its character count for bridging
        const targetSlide = this._reveal.getSlide(targetH, targetV);
        const targetCharCount = targetSlide
            ? (this._slideCharCounts.get(targetSlide) || this._countTextChars(targetSlide))
            : 0;

        this._isAnimating = true;
        this._reveal.getRevealElement().classList.add('diffusion-animating');
        this._encodeSlideNumber();

        const charSpans = this._wrapTextNodes(currentSlide);

        // Start chart exit (CSS blur+fade) in parallel with text encoding.
        // This runs for all slides with charts, even text-free ones.
        const chartExitPromise = this._runChartExit(currentSlide);

        if (charSpans.length === 0) {
            // No text to encode — wait for chart exit, then swap
            chartExitPromise.then(() => {
                currentSlide.classList.remove('diffusion-exiting');
                this._restoreDOM(currentSlide);
                this._isAnimating = false;
                this._reveal.getRevealElement().classList.remove('diffusion-animating');
                this._bypassNext = true;
                this._reveal.slide(targetH, targetV);
            });
            return;
        }

        const currentCount = charSpans.length;
        const K = this._computeSharedCount(currentCount, targetCharCount);

        // Phase 1: Encode text → blocks + chart exit (in parallel)
        // Phase 2: Morph block count from N → K (200ms)
        // Then: slide swap happens at maximum noise
        Promise.all([
            this._animateEncode(charSpans, DIFFUSION_CONFIG.exitDuration),
            chartExitPromise,
        ]).then(() => {
            // Now all spans are blocks — morph to shared count K
            return this._morphBlockCount(currentSlide, charSpans, currentCount, K, DIFFUSION_CONFIG.morphDuration);
        }).then(() => {
            // Store K for the entrance side to pick up
            this._bridgeCount = K;
            this._bridgeTargetCount = targetCharCount;

            currentSlide.classList.remove('diffusion-exiting');
            this._restoreDOM(currentSlide);
            this._isAnimating = false;
            this._reveal.getRevealElement().classList.remove('diffusion-animating');
            this._bypassNext = true;
            this._reveal.slide(targetH, targetV);
        });
    }

    _onSlideChanged(event) {
        const slide = event.currentSlide;
        if (!slide) return;

        // Skip in overview
        if (this._reveal.isOverview()) return;

        // Pass bridge info to the entrance (use typeof check to allow 0 as valid)
        const bridgeCount = (this._bridgeCount != null) ? this._bridgeCount : null;
        const bridgeTargetCount = (this._bridgeTargetCount != null) ? this._bridgeTargetCount : null;
        this._bridgeCount = null;
        this._bridgeTargetCount = null;

        this._runEntrance(slide, bridgeCount);
    }

    // ─── Entrance Animation ──────────────────────────────────────

    _runEntrance(slide, bridgeCount) {
        // Add entering class for CSS-based fades on math/charts/images
        slide.classList.add('diffusion-entering');
        this._animateSlideNumberEntrance();

        const charSpans = this._wrapTextNodes(slide);

        if (charSpans.length === 0) {
            // Nothing to animate, just trigger the CSS fades
            this._triggerMediaFades(slide);
            setTimeout(() => {
                slide.classList.remove('diffusion-entering');
            }, DIFFUSION_CONFIG.enterDuration);
            return;
        }

        // Set initial state: all real chars show as random blocks
        for (const span of charSpans) {
            span.textContent = BLOCK_CHARS[Math.floor(Math.random() * BLOCK_CHARS.length)];
        }

        this._isAnimating = true;
        this._reveal.getRevealElement().classList.add('diffusion-animating');

        const actualCount = charSpans.length; // M — the real character count

        // Set up initial visual state to match bridge count K, then morph to M
        let morphPromise = Promise.resolve();

        if (bridgeCount !== null && bridgeCount !== actualCount) {
            if (bridgeCount > actualCount) {
                // K > M: the exit left more blocks than this slide has chars.
                // Insert (K - M) extra morph blocks to match the exit state,
                // then the morph will REMOVE them down to M.
                const extraCount = bridgeCount - actualCount;
                this._insertInitialMorphBlocks(slide, charSpans, extraCount);
                // Now the DOM has K blocks total. Morph removes extras.
                morphPromise = this._morphBlockCount(
                    slide, charSpans, bridgeCount, actualCount,
                    DIFFUSION_CONFIG.morphDuration
                );
            } else {
                // K < M: the exit left fewer blocks than this slide needs.
                // Hide (M - K) real spans initially, then morph reveals them.
                const hideCount = actualCount - bridgeCount;
                const hidden = this._hideInitialSpans(charSpans, hideCount);
                // Now K spans are visible. Morph adds the hidden ones back.
                morphPromise = this._morphRevealHidden(
                    slide, hidden, DIFFUSION_CONFIG.morphDuration
                );
            }
        }

        morphPromise.then(() => {
            // Re-query spans since morphing may have added/removed some
            const currentSpans = Array.from(slide.querySelectorAll('.diff-char'));

            const chartsDone = this._triggerMediaFades(slide);

            return this._animateResolve(currentSpans, DIFFUSION_CONFIG.enterDuration)
                .then(() => chartsDone);
        }).then(() => {
            this._restoreDOM(slide);
            slide.classList.remove('diffusion-entering');
            slide.classList.remove('diffusion-math-visible');
            slide.classList.remove('diffusion-charts-visible');
            slide.classList.remove('diffusion-media-visible');
            this._isAnimating = false;
            this._reveal.getRevealElement().classList.remove('diffusion-animating');
        });
    }

    /**
     * Insert extra morph-block spans into the slide to reach a higher initial
     * block count (K > M). These blocks are distributed evenly across the
     * existing spans and will be removed during the morph phase.
     */
    _insertInitialMorphBlocks(slide, charSpans, count) {
        if (charSpans.length === 0 || count === 0) return;

        const stride = Math.max(1, Math.floor(charSpans.length / count));

        for (let i = 0; i < count; i++) {
            const anchorIdx = Math.min(
                (i * stride) % charSpans.length,
                charSpans.length - 1
            );
            const anchor = charSpans[anchorIdx];

            const span = document.createElement('span');
            span.className = 'diff-char';
            span.dataset.original = BLOCK_CHARS[0]; // mark as morph block
            span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];

            if (anchor.nextSibling) {
                anchor.parentNode.insertBefore(span, anchor.nextSibling);
            } else {
                anchor.parentNode.appendChild(span);
            }
        }
    }

    /**
     * Hide a subset of real char spans initially (for K < M case).
     * Returns the array of hidden spans so the morph can reveal them.
     * Hidden spans are distributed evenly across the layout.
     */
    _hideInitialSpans(charSpans, count) {
        const stride = Math.max(1, Math.floor(charSpans.length / count));
        const hidden = [];
        const hideSet = new Set();

        for (let i = 0; i < count; i++) {
            let idx = (i * stride) % charSpans.length;
            while (hideSet.has(idx) && idx < charSpans.length - 1) idx++;
            if (!hideSet.has(idx)) {
                hideSet.add(idx);
                charSpans[idx].classList.add('diff-morph-in'); // opacity: 0
                hidden.push(charSpans[idx]);
            }
        }

        return hidden;
    }

    /**
     * Reveal previously hidden spans with staggered fade-in (for K < M entrance).
     * During the reveal, visible blocks continue jittering for noise texture.
     */
    _morphRevealHidden(slide, hiddenSpans, duration) {
        return new Promise((resolve) => {
            if (hiddenSpans.length === 0) { resolve(); return; }

            const staggerTotal = Math.min(
                duration * 0.7,
                hiddenSpans.length * DIFFUSION_CONFIG.morphStaggerMs
            );
            const perSpanDelay = hiddenSpans.length > 1
                ? staggerTotal / (hiddenSpans.length - 1) : 0;

            const allVisible = Array.from(slide.querySelectorAll('.diff-char:not(.diff-morph-in)'));
            const startTime = performance.now();

            const tick = (now) => {
                const elapsed = now - startTime;

                // Jitter visible blocks
                for (const span of allVisible) {
                    if (Math.random() < DIFFUSION_CONFIG.jitterRate) {
                        span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                }

                // Reveal hidden spans based on stagger timing
                for (let i = 0; i < hiddenSpans.length; i++) {
                    const spanDelay = i * perSpanDelay;
                    if (elapsed >= spanDelay) {
                        const span = hiddenSpans[i];
                        if (span.classList.contains('diff-morph-in')) {
                            span.classList.remove('diff-morph-in');
                            span.classList.add('diff-morph-visible');
                        }
                    }
                }

                if (elapsed < duration) {
                    requestAnimationFrame(tick);
                } else {
                    // Clean up classes
                    for (const span of hiddenSpans) {
                        span.classList.remove('diff-morph-in', 'diff-morph-visible');
                    }
                    resolve();
                }
            };

            requestAnimationFrame(tick);
        });
    }

    _triggerMediaFades(slide) {
        setTimeout(() => {
            slide.classList.add('diffusion-media-visible');
        }, DIFFUSION_CONFIG.mediaFadeDelay);

        // Charts: add the visibility class after a short delay.
        // The CSS transition handles the actual blur+fade+scale animation.
        // Returns a promise that resolves after the CSS transition completes
        // so _restoreDOM waits for charts to finish animating in.
        const chartsDone = new Promise(resolve => {
            const charts = slide.querySelectorAll('.chart-container');
            if (charts.length === 0) {
                setTimeout(() => {
                    slide.classList.add('diffusion-charts-visible');
                    resolve();
                }, DIFFUSION_CONFIG.chartFadeDelay);
                return;
            }

            setTimeout(() => {
                slide.classList.add('diffusion-charts-visible');
                // Wait for the longest CSS transition to complete (600ms blur)
                setTimeout(resolve, 620);
            }, DIFFUSION_CONFIG.chartFadeDelay);
        });

        setTimeout(() => {
            slide.classList.add('diffusion-math-visible');
        }, DIFFUSION_CONFIG.mathFadeDelay);

        return chartsDone;
    }

    // ─── Chart Exit (CSS-driven) ──────────────────────────────────

    /**
     * Run chart exit animation on a slide via CSS classes.
     *
     * Adds 'diffusion-exiting' which triggers a blur + fade + scale-down
     * transition in CSS, then resolves after the CSS transition completes.
     * No canvas overlays, no Plotly.toImage — pure CSS, GPU-accelerated.
     */
    _runChartExit(slide) {
        const charts = slide.querySelectorAll('.chart-container');
        if (charts.length === 0) return Promise.resolve();

        slide.classList.add('diffusion-exiting');

        return new Promise(resolve => {
            setTimeout(() => {
                resolve();
            }, DIFFUSION_CONFIG.chartExitDuration);
        });
    }

    // ─── Block Count Morphing ────────────────────────────────────

    /**
     * Smoothly transition the number of block-character spans on a slide.
     *
     * If targetCount > currentCount: insert new block spans distributed
     * evenly across the existing layout (not just appended at the end).
     *
     * If targetCount < currentCount: fade out and remove spans distributed
     * evenly across the layout.
     *
     * During morphing, surviving blocks continue their jitter animation
     * to maintain the noise texture.
     *
     * @param {Element} slide - The slide element
     * @param {HTMLSpanElement[]} charSpans - Current array of .diff-char spans
     * @param {number} fromCount - Starting block count
     * @param {number} toCount - Target block count
     * @param {number} duration - Total morph duration in ms
     * @returns {Promise} Resolves when morph is complete
     */
    _morphBlockCount(slide, charSpans, fromCount, toCount, duration) {
        return new Promise((resolve) => {
            const diff = toCount - fromCount;

            if (diff === 0) {
                resolve();
                return;
            }

            // Get current spans from the DOM (they may differ from the passed array
            // if we're on the entrance side where we started with a different count)
            let liveSpans = Array.from(slide.querySelectorAll('.diff-char'));

            if (diff > 0) {
                // ADDING blocks: distribute new spans evenly among existing ones
                this._morphAddBlocks(slide, liveSpans, diff, duration).then(resolve);
            } else {
                // REMOVING blocks: dissolve spans evenly from the layout
                this._morphRemoveBlocks(slide, liveSpans, Math.abs(diff), duration).then(resolve);
            }
        });
    }

    /**
     * Add `count` new block spans distributed evenly across the slide.
     * New blocks fade in with a staggered delay to create an organic
     * "growing" feel rather than a sudden appearance.
     */
    _morphAddBlocks(slide, existingSpans, count, duration) {
        return new Promise((resolve) => {
            if (existingSpans.length === 0 || count === 0) {
                resolve();
                return;
            }

            // Compute insertion positions: distribute evenly with stride
            const stride = Math.max(1, Math.floor(existingSpans.length / count));
            const newSpans = [];

            // Create all new spans first, insert them into the DOM
            for (let i = 0; i < count; i++) {
                // Pick an insertion anchor: stride through existing spans,
                // wrapping around if we exceed the array
                const anchorIdx = Math.min(
                    (i * stride) % existingSpans.length,
                    existingSpans.length - 1
                );
                const anchor = existingSpans[anchorIdx];

                const span = document.createElement('span');
                span.className = 'diff-char diff-morph-in';
                span.dataset.original = BLOCK_CHARS[0];
                span.textContent = BLOCK_CHARS[Math.floor(Math.random() * BLOCK_CHARS.length)];

                // Insert after the anchor
                if (anchor.nextSibling) {
                    anchor.parentNode.insertBefore(span, anchor.nextSibling);
                } else {
                    anchor.parentNode.appendChild(span);
                }
                newSpans.push(span);
            }

            // Stagger the fade-in: each new span gets a delay based on its index
            const staggerTotal = Math.min(duration * 0.7, count * DIFFUSION_CONFIG.morphStaggerMs);
            const perSpanDelay = count > 1 ? staggerTotal / (count - 1) : 0;

            const startTime = performance.now();

            // Run jitter on all blocks during morph for continuous noise texture
            const allSpans = Array.from(slide.querySelectorAll('.diff-char'));
            let completed = 0;

            const tick = (now) => {
                const elapsed = now - startTime;

                // Jitter all visible blocks
                for (const span of allSpans) {
                    if (!span.classList.contains('diff-morph-removing') &&
                        Math.random() < DIFFUSION_CONFIG.jitterRate) {
                        span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                }

                // Fade in new spans based on stagger timing
                completed = 0;
                for (let i = 0; i < newSpans.length; i++) {
                    const spanDelay = i * perSpanDelay;
                    if (elapsed >= spanDelay) {
                        const span = newSpans[i];
                        if (span.classList.contains('diff-morph-in')) {
                            span.classList.remove('diff-morph-in');
                            span.classList.add('diff-morph-visible');
                        }
                        completed++;
                    }
                }

                if (elapsed < duration) {
                    requestAnimationFrame(tick);
                } else {
                    // Ensure all are visible and clean up classes
                    for (const span of newSpans) {
                        span.classList.remove('diff-morph-in', 'diff-morph-visible');
                    }
                    resolve();
                }
            };

            requestAnimationFrame(tick);
        });
    }

    /**
     * Remove `count` block spans distributed evenly across the slide.
     * Removed blocks fade out with staggered timing, then are removed
     * from the DOM.
     */
    _morphRemoveBlocks(slide, existingSpans, count, duration) {
        return new Promise((resolve) => {
            if (existingSpans.length === 0 || count === 0) {
                resolve();
                return;
            }

            // Clamp: can't remove more than we have
            count = Math.min(count, existingSpans.length);

            // Prefer removing morph-inserted blocks (data-original is a block char)
            // over real character spans. This matters on the entrance side where we
            // inserted extra blocks to match K, then need to remove them.
            const morphBlocks = [];
            const realBlocks = [];
            for (let i = 0; i < existingSpans.length; i++) {
                if (BLOCK_CHARS.includes(existingSpans[i].dataset.original)) {
                    morphBlocks.push(i);
                } else {
                    realBlocks.push(i);
                }
            }

            const toRemove = [];
            const removeSet = new Set();

            // First, remove morph blocks (distributed evenly among them)
            const morphToRemove = Math.min(count, morphBlocks.length);
            if (morphToRemove > 0) {
                const morphStride = Math.max(1, Math.floor(morphBlocks.length / morphToRemove));
                for (let i = 0; i < morphToRemove; i++) {
                    const idx = morphBlocks[Math.min(i * morphStride, morphBlocks.length - 1)];
                    if (!removeSet.has(idx)) {
                        removeSet.add(idx);
                        toRemove.push(existingSpans[idx]);
                    }
                }
            }

            // If we still need to remove more, take from real blocks
            const remaining = count - toRemove.length;
            if (remaining > 0 && realBlocks.length > 0) {
                const realStride = Math.max(1, Math.floor(realBlocks.length / remaining));
                for (let i = 0; i < remaining; i++) {
                    const idx = realBlocks[Math.min(i * realStride, realBlocks.length - 1)];
                    let actualIdx = idx;
                    while (removeSet.has(actualIdx) && actualIdx < existingSpans.length - 1) {
                        actualIdx++;
                    }
                    if (!removeSet.has(actualIdx)) {
                        removeSet.add(actualIdx);
                        toRemove.push(existingSpans[actualIdx]);
                    }
                }
            }

            // Stagger the fade-out
            const staggerTotal = Math.min(duration * 0.7, toRemove.length * DIFFUSION_CONFIG.morphStaggerMs);
            const perSpanDelay = toRemove.length > 1 ? staggerTotal / (toRemove.length - 1) : 0;

            const startTime = performance.now();
            const removeSetSpans = new Set(toRemove);
            const survivors = existingSpans.filter(s => !removeSetSpans.has(s));
            let removed = 0;

            const tick = (now) => {
                const elapsed = now - startTime;

                // Jitter surviving blocks
                for (const span of survivors) {
                    if (Math.random() < DIFFUSION_CONFIG.jitterRate) {
                        span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                }

                // Fade out and remove targeted spans based on stagger timing
                for (let i = 0; i < toRemove.length; i++) {
                    const span = toRemove[i];
                    if (!span.parentNode) continue; // already removed

                    const spanDelay = i * perSpanDelay;
                    if (elapsed >= spanDelay && !span.classList.contains('diff-morph-removing')) {
                        span.classList.add('diff-morph-removing');
                    }

                    // Remove from DOM after the CSS fade-out (60ms)
                    if (elapsed >= spanDelay + 60 && span.parentNode) {
                        span.parentNode.removeChild(span);
                        removed++;
                    }
                }

                if (elapsed < duration) {
                    requestAnimationFrame(tick);
                } else {
                    // Cleanup: remove any stragglers
                    for (const span of toRemove) {
                        if (span.parentNode) {
                            span.parentNode.removeChild(span);
                        }
                    }
                    resolve();
                }
            };

            requestAnimationFrame(tick);
        });
    }

    // ─── Slide Number Diffusion ─────────────────────────────────

    /**
     * Instantly replace slide-number text with block characters.
     * Called at the start of the exit phase so the number joins
     * the visual noise while the slide content encodes.
     */
    _encodeSlideNumber() {
        const el = document.querySelector('.reveal .slide-number');
        if (!el) return;

        if (this._slideNumberAnimFrame) {
            cancelAnimationFrame(this._slideNumberAnimFrame);
            this._slideNumberAnimFrame = null;
        }

        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        const textNodes = [];
        let node;
        while ((node = walker.nextNode())) {
            if (node.textContent.trim()) textNodes.push(node);
        }

        for (const textNode of textNodes) {
            let encoded = '';
            for (const ch of textNode.textContent) {
                if (ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r') {
                    encoded += ch;
                } else {
                    encoded += BLOCK_CHARS[Math.floor(Math.random() * 3)];
                }
            }
            textNode.textContent = encoded;
        }
    }

    /**
     * Animate the slide-number text resolving from block characters,
     * synced with the main entrance animation timing.
     */
    _animateSlideNumberEntrance() {
        const el = document.querySelector('.reveal .slide-number');
        if (!el) return;

        if (this._slideNumberAnimFrame) {
            cancelAnimationFrame(this._slideNumberAnimFrame);
            this._slideNumberAnimFrame = null;
            // Clean up leftover diff-char spans from the cancelled animation
            // so we don't end up wrapping spans-inside-spans or preserving
            // stale block characters as "original" text.
            const staleSpans = el.querySelectorAll('.diff-char');
            for (const span of staleSpans) {
                if (span.parentNode) {
                    span.parentNode.replaceChild(
                        document.createTextNode(span.dataset.original || span.textContent),
                        span
                    );
                }
            }
            el.normalize();
        }

        // Collect text nodes
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        const textNodes = [];
        let node;
        while ((node = walker.nextNode())) {
            if (node.textContent.trim()) textNodes.push(node);
        }
        if (textNodes.length === 0) return;

        // Wrap each non-whitespace char in a span, starting as blocks.
        //
        // NOTE: We intentionally do NOT snapshot el.innerHTML for later
        // restoration. The previous approach (`el.innerHTML = originalHTML`)
        // caused a bug: _encodeSlideNumber() replaces text with block chars
        // during the exit phase, and Reveal.js may not have updated the
        // slide-number element by the time this entrance method captures
        // the snapshot — so the "original" HTML was actually the encoded
        // block-char state, and restoring it left blocks visible as the
        // final rendered text. Instead, after the animation resolves every
        // span to its dataset.original character, we replace each span with
        // a plain text node and normalize the parent.
        const allSpans = [];
        for (const textNode of textNodes) {
            const text = textNode.textContent;
            const frag = document.createDocumentFragment();
            for (const ch of text) {
                if (ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r') {
                    frag.appendChild(document.createTextNode(ch));
                } else {
                    const span = document.createElement('span');
                    span.className = 'diff-char';
                    span.dataset.original = ch;
                    span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    frag.appendChild(span);
                    allSpans.push(span);
                }
            }
            textNode.parentNode.replaceChild(frag, textNode);
        }

        if (allSpans.length === 0) {
            return;
        }

        const duration = DIFFUSION_CONFIG.enterDuration * 1.4;
        const stepMs = DIFFUSION_CONFIG.stepMs;
        const jitter = 0.25;

        const schedule = allSpans.map(span => ({
            span,
            resolveAt: Math.random() * (duration - stepMs * 3),
            stage: 0,
            original: span.dataset.original,
        }));

        const startTime = performance.now();
        let resolved = 0;

        const tick = (now) => {
            const elapsed = now - startTime;

            for (const item of schedule) {
                if (item.stage >= 3) continue;

                const charElapsed = elapsed - item.resolveAt;
                if (charElapsed < 0) {
                    if (Math.random() < jitter) {
                        item.span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                    continue;
                }

                const newStage = Math.min(3, Math.floor(charElapsed / stepMs));
                if (newStage > item.stage) {
                    item.stage = newStage;
                    if (newStage === 3) {
                        item.span.textContent = item.original;
                        resolved++;
                    } else {
                        item.span.textContent = BLOCK_CHARS[newStage];
                    }
                }
            }

            if (resolved < allSpans.length) {
                this._slideNumberAnimFrame = requestAnimationFrame(tick);
            } else {
                // Replace each diff-char span with a plain text node
                // containing the resolved original character. This avoids
                // relying on a stale innerHTML snapshot that may contain
                // block characters from the prior _encodeSlideNumber() call.
                for (const item of schedule) {
                    const span = item.span;
                    if (span.parentNode) {
                        span.parentNode.replaceChild(
                            document.createTextNode(item.original),
                            span
                        );
                    }
                }
                // Merge adjacent text nodes for clean DOM
                el.normalize();
                this._slideNumberAnimFrame = null;
            }
        };

        this._slideNumberAnimFrame = requestAnimationFrame(tick);
    }

    // ─── DOM Walking ─────────────────────────────────────────────

    _shouldSkip(el) {
        if (!el) return true;
        return el.closest(
            '.katex, .katex-display, .katex-html, ' +
            '.chart-container, [id^="chart-"], .plotly, .js-plotly-plot, ' +
            'img, svg, video, canvas, ' +
            'aside.notes, ' +
            'script, style, noscript, ' +
            '.MathJax, .MathJax_Display'
        ) !== null;
    }

    _collectTextNodes(root) {
        const nodes = [];
        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: (node) => {
                    if (this._shouldSkip(node.parentElement)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    if (!node.textContent.trim()) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );
        let node;
        while ((node = walker.nextNode())) {
            nodes.push(node);
        }
        return nodes;
    }

    _wrapTextNodes(slide) {
        // Snapshot the slide's DOM *before* mutation.  Only take a fresh
        // snapshot when the slide isn't already wrapped (i.e. no existing
        // snapshot or the previous one was consumed by _restoreDOM).
        if (!this._domSnapshots.has(slide)) {
            this._domSnapshots.set(slide, slide.innerHTML);
        }

        const textNodes = this._collectTextNodes(slide);
        const allSpans = [];

        for (const textNode of textNodes) {
            const text = textNode.textContent;
            const frag = document.createDocumentFragment();

            for (let i = 0; i < text.length; i++) {
                const ch = text[i];
                if (ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r') {
                    frag.appendChild(document.createTextNode(ch));
                } else {
                    const span = document.createElement('span');
                    span.className = 'diff-char';
                    span.dataset.original = ch;
                    span.textContent = ch;
                    frag.appendChild(span);
                    allSpans.push(span);
                }
            }

            textNode.parentNode.replaceChild(frag, textNode);
        }

        return allSpans;
    }

    _restoreDOM(slide) {
        // Restore from the snapshot taken before _wrapTextNodes mutated the
        // DOM.  This is the only reliable way to undo all mutations because
        // the morph phase may have removed real-character spans (when the
        // exit morph contracts N blocks down to K < N), permanently losing
        // those characters from the live tree.  Replaying the snapshot
        // guarantees the slide is returned to its exact original state.
        const snapshot = this._domSnapshots.get(slide);
        if (snapshot != null) {
            slide.innerHTML = snapshot;
            this._domSnapshots.delete(slide);
        } else {
            // Fallback: no snapshot (should not happen in normal flow).
            // Do a best-effort span-by-span restoration.
            const morphSpans = slide.querySelectorAll('.diff-char');
            for (const span of morphSpans) {
                const original = span.dataset.original;
                if (BLOCK_CHARS.includes(original)) {
                    if (span.parentNode) {
                        span.parentNode.removeChild(span);
                    }
                } else {
                    const textNode = document.createTextNode(original);
                    span.parentNode.replaceChild(textNode, span);
                }
            }
            slide.normalize();
        }
    }

    // ─── Animation ───────────────────────────────────────────────

    _animateResolve(charSpans, duration) {
        return new Promise((resolve) => {
            if (charSpans.length === 0) { resolve(); return; }

            const stepMs = DIFFUSION_CONFIG.stepMs;
            const jitter = DIFFUSION_CONFIG.jitterRate;

            // Separate real chars (to resolve) from morph blocks (to dissolve)
            const realChars = [];
            const morphBlocks = [];

            for (const span of charSpans) {
                if (BLOCK_CHARS.includes(span.dataset.original)) {
                    morphBlocks.push(span);
                } else {
                    realChars.push(span);
                }
            }

            // Each real char gets a random time to start resolving
            const schedule = realChars.map((span) => ({
                span,
                resolveAt: Math.random() * (duration - stepMs * 3),
                stage: 0, // 0=blocks, 1=dark shade, 2=light shade, 3=resolved
                original: span.dataset.original,
            }));

            // Morph blocks dissolve during the first half of the resolve
            const morphSchedule = morphBlocks.map((span, i) => ({
                span,
                dissolveAt: Math.random() * (duration * 0.5),
                dissolved: false,
            }));

            // Initial state: random blocks for all
            for (const item of schedule) {
                item.span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
            }

            const startTime = performance.now();
            let resolved = 0;

            const tick = (now) => {
                const elapsed = now - startTime;

                // Dissolve morph blocks
                for (const item of morphSchedule) {
                    if (item.dissolved) continue;
                    if (elapsed >= item.dissolveAt) {
                        item.span.classList.add('diff-morph-removing');
                        // Remove from DOM shortly after
                        if (elapsed >= item.dissolveAt + 60) {
                            if (item.span.parentNode) {
                                item.span.parentNode.removeChild(item.span);
                            }
                            item.dissolved = true;
                        }
                    } else if (Math.random() < jitter) {
                        item.span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                }

                // Resolve real chars
                for (const item of schedule) {
                    if (item.stage >= 3) continue;

                    const charElapsed = elapsed - item.resolveAt;
                    if (charElapsed < 0) {
                        // Still waiting — random jitter between block chars
                        if (Math.random() < jitter) {
                            item.span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                        }
                        continue;
                    }

                    const newStage = Math.min(3, Math.floor(charElapsed / stepMs));
                    if (newStage > item.stage) {
                        item.stage = newStage;
                        if (newStage === 3) {
                            item.span.textContent = item.original;
                            resolved++;
                        } else {
                            // Resolving: █(0) -> ▓(1) -> ░(2) -> char(3)
                            item.span.textContent = BLOCK_CHARS[newStage];
                        }
                    }
                }

                if (resolved < realChars.length) {
                    requestAnimationFrame(tick);
                } else {
                    // Clean up any remaining morph blocks
                    for (const item of morphSchedule) {
                        if (!item.dissolved && item.span.parentNode) {
                            item.span.parentNode.removeChild(item.span);
                        }
                    }
                    resolve();
                }
            };

            requestAnimationFrame(tick);
        });
    }

    _animateEncode(charSpans, duration) {
        return new Promise((resolve) => {
            if (charSpans.length === 0) { resolve(); return; }

            const stepMs = DIFFUSION_CONFIG.stepMs;
            const jitter = DIFFUSION_CONFIG.jitterRate;

            // Each char gets a random time to start encoding
            const schedule = charSpans.map((span) => ({
                span,
                encodeAt: Math.random() * (duration - stepMs * 3),
                stage: 0, // 0=clear, 1=light, 2=dark, 3=full block
                original: span.dataset.original,
            }));

            const startTime = performance.now();
            let encoded = 0;

            const tick = (now) => {
                const elapsed = now - startTime;

                for (const item of schedule) {
                    if (item.stage >= 3) continue;

                    const charElapsed = elapsed - item.encodeAt;
                    if (charElapsed < 0) continue;

                    const newStage = Math.min(3, Math.floor(charElapsed / stepMs));
                    if (newStage > item.stage) {
                        item.stage = newStage;
                        if (newStage === 3) {
                            item.span.textContent = BLOCK_CHARS[0]; // █
                            encoded++;
                        } else {
                            // Encoding: char(0) -> ░(1) -> ▓(2) -> █(3)
                            item.span.textContent = BLOCK_CHARS[2 - (newStage - 1)];
                        }
                    }
                }

                // Add jitter to already-encoded chars
                for (const item of schedule) {
                    if (item.stage >= 3 && Math.random() < jitter * 0.5) {
                        item.span.textContent = BLOCK_CHARS[Math.floor(Math.random() * 3)];
                    }
                }

                if (encoded < charSpans.length) {
                    requestAnimationFrame(tick);
                } else {
                    resolve();
                }
            };

            requestAnimationFrame(tick);
        });
    }
}

// Initialize when Reveal.js is ready
if (typeof Reveal !== 'undefined') {
    Reveal.on('ready', () => {
        const engine = new DiffusionEngine(Reveal);
        engine.init();
        window.__diffusionEngine = engine;
    });
}
