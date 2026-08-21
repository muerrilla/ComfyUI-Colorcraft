(function () {
    const CHECKBOX_SELECTOR = '.colorcraft-tab-group .colorcraft-active input[type="checkbox"]';
    const TARGETED_SELECTOR = '.colorcraft-tab-group .colorcraft-targeted input[type="checkbox"]';
    const LABEL_SELECTOR = '.colorcraft-tab-group .colorcraft-combo-label textarea, .colorcraft-tab-group .colorcraft-combo-label input';

    function findTabId(el) {
        while (el && el !== document.body) {
            if (el.classList.contains('colorcraft-tab') && el.id && /^component-\d+$/.test(el.id)) {
                return el.id;
            }
            el = el.parentElement;
        }
        return null;
    }

    function syncButton(checkbox) {
        const tabId = findTabId(checkbox);
        if (!tabId) return;
        const button = gradioApp().getElementById(`${tabId}-button`);
        if (!button) return;
        button.classList.toggle('colorcraft-tab-active', checkbox.checked);
    }

    // A modifier's own "is a mask/combo currently aimed at me" state --
    // independent of its Active checkbox above. Same sync mechanism,
    // separate class so the two signals style distinctly.
    function syncTargetedButton(checkbox) {
        const tabId = findTabId(checkbox);
        if (!tabId) return;
        const button = gradioApp().getElementById(`${tabId}-button`);
        if (!button) return;
        button.classList.toggle('colorcraft-tab-targeted', checkbox.checked);
    }

    // Combo tabs: the label textbox's value becomes the tab button's
    // visible text via a data-attribute + CSS overlay (see style.css),
    // e.g. "M3 ⊕ M5" once both Mask A and B are set, its own tag
    // otherwise. Deliberately NOT writing button.textContent directly --
    // that's Gradio's Svelte-managed state, and an external write gets
    // fought over on the next re-render (Svelte reconciles it back, that
    // mutation re-triggers the MutationObserver onUiUpdate runs on, which
    // calls this function again -- a tight infinite loop). A custom
    // data-attribute isn't part of Gradio's bound state, so nothing
    // resets it and nothing loops.
    function syncLabel(textbox) {
        const tabId = findTabId(textbox);
        if (!tabId) return;
        const button = gradioApp().getElementById(`${tabId}-button`);
        if (!button) return;
        if (button.getAttribute('data-colorcraft-label') !== textbox.value) {
            button.setAttribute('data-colorcraft-label', textbox.value);
        }
    }

    function syncAll() {
        gradioApp().querySelectorAll(CHECKBOX_SELECTOR).forEach(syncButton);
        gradioApp().querySelectorAll(TARGETED_SELECTOR).forEach(syncTargetedButton);
        gradioApp().querySelectorAll(LABEL_SELECTOR).forEach(syncLabel);
    }

    // Modifier tabs' Active checkbox fires 'input' on a real click.
    // Mask/combo tabs' active/targeted checkboxes and the combo label
    // textbox are hidden and only ever set programmatically, which fires
    // 'change' instead -- both event types are listened for here.
    document.addEventListener('input', (e) => {
        if (e.target.matches(CHECKBOX_SELECTOR)) syncButton(e.target);
        else if (e.target.matches(TARGETED_SELECTOR)) syncTargetedButton(e.target);
        else if (e.target.matches(LABEL_SELECTOR)) syncLabel(e.target);
    });
    document.addEventListener('change', (e) => {
        if (e.target.matches(CHECKBOX_SELECTOR)) syncButton(e.target);
        else if (e.target.matches(TARGETED_SELECTOR)) syncTargetedButton(e.target);
        else if (e.target.matches(LABEL_SELECTOR)) syncLabel(e.target);
    });

    onUiLoaded(syncAll);
    onUiUpdate(syncAll);
})();

// ---------------------------------------------------------------------------
// Schedule plot (per modifier tab). Separate IIFE from the tab-highlight
// logic above -- different concern, no shared state.
//
// makeScheduleArray is ported near-verbatim from Colorcraft's Comfy-side
// colorcraft.js (makeScheduleArray/getScheduleParams) -- that math has no
// LiteGraph dependency. Everything else here (canvas element, redraw-on-
// input wiring, ResizeObserver, reading the live sampling-steps field) is
// a from-scratch rendering/wiring layer around the ported math, since
// Forge has no LiteGraph canvas-widget registry or render loop.
// ---------------------------------------------------------------------------

(function () {
    const SCHEDULE_PLOT_RESOLUTION = 200;

    function makeScheduleArray(n, p) {
        const lo = Math.min(p.start, p.end);
        const hi = p.end;
        const mid = lo + p.bias * (hi - lo);
        const arr = new Float64Array(n);
        const toIdx = (x) => Math.round(x * (n - 1));
        const startIdx = toIdx(lo);
        const midIdx = toIdx(mid);
        const endIdx = toIdx(hi);

        const ease = (raw) => {
            const r = p.smooth ? 0.5 * (1 - Math.cos(raw * Math.PI)) : raw;
            return p.exponent >= 0 ? Math.pow(r, p.exponent) : 1 - Math.pow(1 - r, 1 / Math.abs(p.exponent));
        };

        const nStart = midIdx - startIdx + 1;
        for (let i = 0; i < nStart; i++) {
            const raw = nStart > 1 ? i / (nStart - 1) : 0;
            const idx = startIdx + i;
            if (idx >= 0 && idx < n) arr[idx] = ease(raw) * (p.amount - p.start_off) + p.start_off;
        }
        const nEnd = endIdx - midIdx + 1;
        for (let i = 0; i < nEnd; i++) {
            const raw = nEnd > 1 ? 1 - i / (nEnd - 1) : 1;
            const idx = midIdx + i;
            if (idx >= 0 && idx < n) arr[idx] = ease(raw) * (p.amount - p.end_off) + p.end_off;
        }
        for (let i = 0; i < Math.min(startIdx, n); i++) arr[i] = p.start_off;
        for (let i = Math.max(endIdx + 1, 0); i < n; i++) arr[i] = p.end_off;

        return { arr, mid };
    }

    function readNumber(el, fallback) {
        if (!el) return fallback;
        const v = parseFloat(el.value);
        return Number.isNaN(v) ? fallback : v;
    }

    function drawSchedulePlot(canvas, p, steps) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, rect.width);
        const h = Math.max(1, rect.height);
        const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        const margin = 5, padding = 2;
        const x0 = margin, x1 = w - margin;
        const plotW = x1 - x0;
        const tickH = 3;
        const labelH = 10;
        const plotY = padding / 2;
        const plotH = h - tickH - labelH;
        const toX = (t) => x0 + t * plotW;
        const toY = (v) => plotY + plotH * (1 - (v + 1) / 2);
        const { arr, mid } = makeScheduleArray(SCHEDULE_PLOT_RESOLUTION, p);
        const stroke = "rgba(255,255,255,0.55)";
        const stepDivisor = Math.max(1, Math.round(steps) - 1);
        ctx.save();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0, plotY, plotW, plotH);
        // Bottom ticks with labels
        ctx.fillStyle = stroke;
        ctx.font = `${labelH - 1}px sans-serif`;
        ctx.textBaseline = 'top';
        ctx.textAlign = 'center';
        const labelEvery = stepDivisor > 38 ? 5 : stepDivisor > 18 ? 2 : 1;
        for (let i = 0; i <= stepDivisor; i++) {
            const xx = toX(i / stepDivisor);
            ctx.beginPath(); ctx.moveTo(xx, plotY + plotH); ctx.lineTo(xx, plotY + plotH + tickH); ctx.stroke();
            if (i % labelEvery === 0) ctx.fillText(String(i), xx, plotY + plotH + tickH + padding);
        }
        ctx.setLineDash([1, 3]);
        ctx.beginPath(); ctx.moveTo(x0, toY(0)); ctx.lineTo(x1, toY(0)); ctx.stroke();  
        ctx.setLineDash([]);  
        ctx.lineWidth = .5;
        if (p.exponent !== 0) {
            ctx.beginPath(); ctx.moveTo(toX(mid), plotY); ctx.lineTo(toX(mid), plotY + plotH); ctx.stroke();
        }              
        ctx.beginPath(); ctx.moveTo(toX(p.start), plotY); ctx.lineTo(toX(p.start), plotY + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(toX(p.end), plotY); ctx.lineTo(toX(p.end), plotY + plotH); ctx.stroke();
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0, plotY, plotW, plotH);
        ctx.clip();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i < arr.length; i++) {
            const xx = toX(i / (arr.length - 1));
            const yy = toY(arr[i]);
            if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.stroke();
        ctx.restore();
        ctx.restore();
    }

    // Field lookup is scoped to the tab element the canvas already lives
    // in, so the elem_classes only need to be unique within one tab, not
    // page-wide.
    //
    // Each Gradio Slider renders a paired range+number input, and
    // dragging one does NOT fire an 'input' event on the other -- so
    // both need to be watched, even though either alone is enough to
    // read the current value from.
    function findScheduleFields(tabEl) {
        function pair(cls) {
            const range = tabEl.querySelector(`.${cls} input[type=range]`);
            const number = tabEl.querySelector(`.${cls} input[type=number]`);
            return { readEl: number || range, watchEls: [range, number].filter(Boolean) };
        }
        return {
            strength: pair('colorcraft-strength'),
            start: pair('colorcraft-start'),
            end: pair('colorcraft-end'),
            bias: pair('colorcraft-bias'),
            exponent: pair('colorcraft-exponent'),
            startOff: pair('colorcraft-start-off'),
            endOff: pair('colorcraft-end-off'),
            smoothEl: tabEl.querySelector('.colorcraft-smooth input[type=checkbox]'),
            advancedEl: tabEl.querySelector('.colorcraft-advanced-toggle input[type=checkbox]'),
        };
    }

    // Not watched for live changes -- read once at redraw time, triggered
    // only by the schedule sliders changing. A brief tick-mark mismatch
    // if steps changes without touching a schedule slider is an
    // acceptable tradeoff against redrawing every plot on every drag.
    function readSteps(tabEl) {
        const accordionRoot = tabEl.closest('[id$="_colorcraft_accordion"]');
        const isImg2img = accordionRoot && accordionRoot.id.startsWith('script_img2img');
        const stepsSelector = isImg2img ? '#img2img_steps input[type=number]' : '#txt2img_steps input[type=number]';
        return readNumber(gradioApp().querySelector(stepsSelector), 20);
    }

    const wiredSchedulePlots = new Map(); // canvas -> redraw function
    let stepsWatcherWired = false;

    function setupStepsWatcher() {
        if (stepsWatcherWired) return;
        const els = [
            ...gradioApp().querySelectorAll('#txt2img_steps input'),
            ...gradioApp().querySelectorAll('#img2img_steps input'),
        ];
        if (els.length === 0) return; // not rendered yet -- retry next onUiUpdate
        const redrawAll = () => wiredSchedulePlots.forEach((redraw) => redraw());
        els.forEach((el) => {
            el.addEventListener('input', redrawAll);
            el.addEventListener('change', redrawAll);
        });
        stepsWatcherWired = true;
    }

    function setupSchedulePlots() {
        setupStepsWatcher();
        gradioApp().querySelectorAll('canvas.colorcraft-schedule-plot').forEach((canvas) => {
            if (canvas.dataset.ccWired) return;
            const tabEl = canvas.closest('.colorcraft-tab');
            if (!tabEl) return;
            const f = findScheduleFields(tabEl);
            const watched = [
                ...f.strength.watchEls, ...f.start.watchEls, ...f.end.watchEls,
                ...f.bias.watchEls, ...f.exponent.watchEls, ...f.startOff.watchEls, ...f.endOff.watchEls,
                f.smoothEl, f.advancedEl,
            ].filter(Boolean);
            if (watched.length === 0) return; // fields not rendered yet -- retry next onUiUpdate

            function redraw() {
                const advanced = f.advancedEl ? f.advancedEl.checked : false;
                drawSchedulePlot(canvas, {
                    amount: readNumber(f.strength.readEl, 1),
                    start: readNumber(f.start.readEl, 0.5),
                    end: readNumber(f.end.readEl, 0.75),
                    bias: advanced ? readNumber(f.bias.readEl, 0.5) : 0.5,
                    exponent: advanced ? readNumber(f.exponent.readEl, 0) : 0,
                    start_off: advanced ? readNumber(f.startOff.readEl, 0) : 0,
                    end_off: advanced ? readNumber(f.endOff.readEl, 0) : 0,
                    smooth: f.smoothEl ? f.smoothEl.checked : true,
                }, readSteps(tabEl));
            }

            watched.forEach((el) => el.addEventListener('input', redraw));
            canvas.dataset.ccWired = "1";
            wiredSchedulePlots.set(canvas, redraw);
            new ResizeObserver(redraw).observe(canvas);
            redraw();
        });
        // Re-run every already-known plot too, not just newly-discovered
        // ones -- reset (and any other server-pushed value change) never
        // fires input/change, but this setup pass itself runs on every
        // onUiUpdate, which DOES fire reliably after such a change.
        wiredSchedulePlots.forEach((redraw) => redraw());
    }

    onUiLoaded(setupSchedulePlots);
    onUiUpdate(setupSchedulePlots);
})();

// ---------------------------------------------------------------------------
// Mask plot (per mask leaf tab).
//
// maskShape/hueMaskValue/wrapAngle/axisGradientStops/LINEAR_AXIS_COLORS are
// ported near-verbatim from Colorcraft's Comfy-side colorcraft.js -- that
// math has no LiteGraph dependency. Verified numerically identical to
// Python's real _mask_shape and the hue-wrap formula in lib_colorcraft/
// masking.py before being wired to anything live.
// ---------------------------------------------------------------------------

(function () {
    const HARDNESS_GAIN = 5.0;
    const MASK_PLOT_RESOLUTION = 200;
    const MASK_PLOT_XMIN = -1.1;
    const MASK_PLOT_XMAX = 1.1;
    const MASK_PLOT_XTICKS = [-1, -0.5, 0, 0.5, 1];
    const HUE_PLOT_HEADROOM = 0.35;
    const HUE_PLOT_XTICKS = [
        [-Math.PI, "-\u03c0"], [-Math.PI / 2, "-\u03c0/2"], [0, "0"],
        [Math.PI / 2, "\u03c0/2"], [Math.PI, "\u03c0"],
    ];

    const LINEAR_AXIS_COLORS = {
        temperature: ["hsl(204, 100%, 50%)", "hsl(30, 100%, 50%)"],
        tint: ["hsl(130, 100%, 45%)", "hsl(304, 100%, 50%)"],
        "lab-a": ["hsl(155, 100%, 45%)", "hsl(322, 100%, 50%)"],
        "lab-b": ["hsl(225, 100%, 50%)", "hsl(55, 100%, 50%)"],
        "temp+tint": ["hsl(181, 100%, 50%)", "hsl(354, 100%, 50%)"],
        "temp-tint": ["hsl(250, 75%, 60%)", "hsl(75, 100%, 50%)"],
        "lab-a+b": ["hsl(198, 100%, 55%)", "hsl(18, 100%, 50%)"],
        "lab-a-b": ["hsl(95, 100%, 45%)", "hsl(270, 100%, 60%)"],
        saturation: ["hsl(0, 0%, 60%)", "hsl(0, 85%, 50%)"],
    };

    function axisGradientStops(colorLo, colorHi, alpha, withGrey = true) {
        const toAlpha = (c) => c.replace("hsl(", "hsla(").replace(")", `, ${alpha})`);
        if (!withGrey) return [[0, toAlpha(colorLo)], [1, toAlpha(colorHi)]];
        return [[0, toAlpha(colorLo)], [0.5, `hsla(0, 0%, 80%, ${alpha})`], [1, toAlpha(colorHi)]];
    }

    function maskShape(mode, hardness, width, strength, diff) {
        const s = hardness * HARDNESS_GAIN;
        const w = width;
        const z = diff * s;
        switch (mode) {
            case "highs": return 1 + strength * (1 / (1 + Math.exp(-z)) - 1);
            case "lows": return 1 + strength * (1 / (1 + Math.exp(z)) - 1);
            case "split": return Math.tanh(z);
            case "range":
            case "protect range": {
                const excess = Math.max(0, Math.abs(diff) - w / 2) * s;
                const g = Math.exp(-0.5 * excess * excess);
                const m = mode === "protect range" ? 1 - g : g;
                return 1 + strength * (m - 1);
            }
            default: return 1;
        }
    }

    function maskValue(mode, center, hardness, width, strength, x) {
        return maskShape(mode, hardness, width, strength, x - center);
    }

    function wrapAngle(a) {
        const twoPi = 2 * Math.PI;
        return ((((a + Math.PI) % twoPi) + twoPi) % twoPi) - Math.PI;
    }

    function hueMaskValue(mode, center, hardness, width, strength, angle) {
        const diff = wrapAngle(angle - center * Math.PI);
        return maskShape(mode, hardness, width, strength, diff / Math.PI);
    }

    function drawGradientStrip(ctx, toX, toY, gradX0, gradX1, plotXmin, plotXmax, stops) {
        ctx.save();
        const gradient = ctx.createLinearGradient(toX(gradX0), toY(0), toX(gradX1), toY(0));
        for (const [t, color] of stops) gradient.addColorStop(t, color);
        ctx.fillStyle = gradient;
        ctx.fillRect(toX(plotXmin), toY(0) + 2, toX(plotXmax) - toX(plotXmin), 8);
        ctx.restore();
    }

    function readNumber(el, fallback) {
        if (!el) return fallback;
        const v = parseFloat(el.value);
        return Number.isNaN(v) ? fallback : v;
    }

    function drawMaskPlot(canvas, p) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, rect.width);
        const h = Math.max(1, rect.height);
        const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const isHue = p.axis === "hue";
        const xmin = isHue ? -Math.PI - HUE_PLOT_HEADROOM : MASK_PLOT_XMIN;
        const xmax = isHue ? Math.PI + HUE_PLOT_HEADROOM : MASK_PLOT_XMAX;

        // Bigger margin than the schedule plot's -- this plot has text tick
        // labels under the x-axis that need room not to clip at the edges.
        const margin = 16;
        const tickLabelSpace = 14;
        const x0 = margin, x1 = w - margin;
        const plotW = x1 - x0;
        const plotY = 2, plotH = h - tickLabelSpace - 4;
        const toX = (v) => x0 + ((v - xmin) / (xmax - xmin)) * plotW;
        const toY = (v) => plotY + plotH * (1 - (v + 1.1) / 2.2);

        const stroke = "rgba(255,255,255,0.55)";

        if (p.axis === "exposure") {
            const alpha = 0.75;
            drawGradientStrip(ctx, toX, toY, -1, 1, xmin, xmax, [
                [0, `hsla(0, 0%, 0%, ${alpha})`],
                [1, `hsla(0, 0%, 100%, ${alpha})`],
            ]);
        }

        if (isHue) {
            const alpha = 0.5;
            const hueAt = (x) => {
                const angle = wrapAngle(x);
                const raw = 220 - ((angle + Math.PI) / (2 * Math.PI)) * 360;
                return ((raw % 360) + 360) % 360;
            };
            const stepsPerPeriod = 24;
            const dx = (xmax - xmin) / stepsPerPeriod;
            const sampleXs = [];
            for (let x = xmin; x < xmax; x += dx) sampleXs.push(x);
            sampleXs.push(xmax);
            const stops = sampleXs.map((x) => [(x - xmin) / (xmax - xmin), `hsla(${hueAt(x)}, 100%, 50%, ${alpha})`]);
            drawGradientStrip(ctx, toX, toY, xmin, xmax, xmin, xmax, stops);
        }

        if (p.axis in LINEAR_AXIS_COLORS) {
            const alpha = 0.6;
            const [colorLo, colorHi] = LINEAR_AXIS_COLORS[p.axis];
            const stops = axisGradientStops(colorLo, colorHi, alpha, p.axis !== "saturation");
            drawGradientStrip(ctx, toX, toY, MASK_PLOT_XMIN, MASK_PLOT_XMAX, xmin, xmax, stops);
        }

        ctx.save();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0, plotY, plotW, plotH);

        ctx.font = "9px sans-serif";
        ctx.fillStyle = stroke;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const ticks = isHue ? HUE_PLOT_XTICKS : MASK_PLOT_XTICKS.map((v) => [v, String(v)]);
        for (const [v, label] of ticks) {
            const xx = toX(v);
            ctx.beginPath();
            ctx.moveTo(xx, plotY + plotH);
            ctx.lineTo(xx, plotY + plotH + 3);
            ctx.stroke();
            ctx.fillText(label, xx, plotY + plotH + 4);
        }

        if (!isHue) {
            // Rough backdrop cue for where real values typically cluster on
            // a +-1-normalized linear axis. Skipped for hue -- no
            // equivalent "typical clustering" shape for a circular axis.
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(toX(xmin), toY(0));
            const bellSigma = (xmax - xmin) * 0.135;
            for (let i = 0; i <= MASK_PLOT_RESOLUTION; i++) {
                const v = xmin + (i / MASK_PLOT_RESOLUTION) * (xmax - xmin);
                const bell = Math.exp(-0.5 * (v / bellSigma) ** 2);
                ctx.lineTo(toX(v), toY(bell));
            }
            ctx.lineTo(toX(xmax), toY(0));
            ctx.closePath();
            ctx.fillStyle = "rgba(255,255,255,0.06)";
            ctx.fill();
            ctx.restore();
        }

        ctx.save();
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(x0, toY(0));
        ctx.lineTo(x1, toY(0));
        ctx.moveTo(toX(0), toY(-1));
        ctx.lineTo(toX(0), toY(1));
        ctx.stroke();
        ctx.restore();

        // Hue's center is circular, so it always has a valid wrapped
        // position on the plot -- unlike a linear axis, where a center far
        // outside the domain is legitimately "off-plot".
        const centerPos = isHue ? wrapAngle(p.center * Math.PI) : p.center;
        if (centerPos >= xmin && centerPos <= xmax) {
            ctx.save();
            ctx.setLineDash([5.5, 5]);
            ctx.beginPath();
            const centerX = toX(centerPos);
            ctx.moveTo(centerX, plotY);
            ctx.lineTo(centerX, plotY + plotH);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        ctx.save();
        ctx.beginPath();
        ctx.rect(x0, plotY, plotW, plotH);
        ctx.clip();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i <= MASK_PLOT_RESOLUTION; i++) {
            const v = xmin + (i / MASK_PLOT_RESOLUTION) * (xmax - xmin);
            const m = isHue
                ? hueMaskValue(p.mode, p.center, p.hardness, p.width, p.strength, v)
                : maskValue(p.mode, p.center, p.hardness, p.width, p.strength, v);
            const xx = toX(v);
            const yy = toY(Math.max(-1.1, Math.min(1.1, m)));
            if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.stroke();
        ctx.restore();

        ctx.restore();
    }

    // Gradio Dropdown exposes its current value via a plain <input>
    // descendant, watched on both 'input' and 'change'.
    function findMaskFields(tabEl) {
        function pair(cls) {
            const range = tabEl.querySelector(`.${cls} input[type=range]`);
            const number = tabEl.querySelector(`.${cls} input[type=number]`);
            return { readEl: number || range, watchEls: [range, number].filter(Boolean) };
        }
        return {
            axisEl: tabEl.querySelector('.colorcraft-mask-axis input'),
            modeEl: tabEl.querySelector('.colorcraft-mask-mode input'),
            center: pair('colorcraft-mask-center'),
            hardness: pair('colorcraft-mask-hardness'),
            width: pair('colorcraft-mask-width'),
            strength: pair('colorcraft-mask-strength'),
        };
    }

    // Neither 'input' nor 'change' fires on a Dropdown's <input> when an
    // option is selected by click -- Gradio's own reactive binding sets
    // the value directly, bypassing the native DOM event path. Rather
    // than guess at exactly where the option list renders (some Gradio
    // versions portal the popup to document.body), this redraws every
    // wired mask plot on any click anywhere -- blunter than ideal, but
    // canvas redraws are cheap and there are at most 10 of these.
    const wiredMaskPlots = new Map();
    document.addEventListener('click', () => {
        wiredMaskPlots.forEach((redraw) => redraw());
    });

    function setupMaskPlots() {
        gradioApp().querySelectorAll('canvas.colorcraft-mask-plot').forEach((canvas) => {
            if (canvas.dataset.ccWired) return;
            const tabEl = canvas.closest('.colorcraft-tab');
            if (!tabEl) return;
            const f = findMaskFields(tabEl);
            const watched = [
                f.axisEl, f.modeEl,
                ...f.center.watchEls, ...f.hardness.watchEls, ...f.width.watchEls, ...f.strength.watchEls,
            ].filter(Boolean);
            if (watched.length === 0) return; // fields not rendered yet -- retry next onUiUpdate

            function redraw() {
                drawMaskPlot(canvas, {
                    axis: f.axisEl ? f.axisEl.value : "exposure",
                    mode: f.modeEl ? f.modeEl.value : "highs",
                    center: readNumber(f.center.readEl, 0),
                    hardness: readNumber(f.hardness.readEl, 1),
                    width: readNumber(f.width.readEl, 0),
                    strength: readNumber(f.strength.readEl, 1),
                });
            }

            watched.forEach((el) => {
                el.addEventListener('input', redraw);
                el.addEventListener('change', redraw);
            });
            canvas.dataset.ccWired = "1";
            wiredMaskPlots.set(canvas, redraw);
            new ResizeObserver(redraw).observe(canvas);
            redraw();
        });
        // Re-run every already-known plot too, not just newly-discovered
        // ones -- same reasoning as setupSchedulePlots: reset (and any
        // other server-pushed value change) never fires input/change, but
        // this setup pass itself runs on every onUiUpdate, which does.
        wiredMaskPlots.forEach((redraw) => redraw());
    }

    onUiLoaded(setupMaskPlots);
    onUiUpdate(setupMaskPlots);
})();

// ---------------------------------------------------------------------------
// "Changed from default" highlighting -- generic across every slider and
// dropdown in both Colorcraft accordions, via the shared wrapper classes
// (.gradio-slider / .colorcraft-dropdown) already on every one of them.
// Excludes checkboxes (the modifier Active checkbox already gets feedback
// via the tab-label highlight).
//
// Default value is captured as whatever's present the first time each
// wrapper is seen. Dropdown selection fires neither 'input' nor 'change'
// at all, so that's caught with a document-level click listener instead.
// Infotext restore updates values correctly but doesn't reliably fire
// input/change either, so every already-wired wrapper's own check() also
// re-runs on every onUiUpdate, which does fire reliably after a restore.
// ---------------------------------------------------------------------------

(function () {
    const wiredChecks = new Map(); // wrapper -> check function, covers sliders and dropdowns alike

    function watchSlider(wrapper) {
        if (wrapper.dataset.ccDefaultCaptured) return;
        const numberEl = wrapper.querySelector('input[type=number]');
        const rangeEl = wrapper.querySelector('input[type=range]');
        if (!numberEl) return; // not rendered yet -- retry next onUiUpdate
        wrapper.dataset.ccDefault = numberEl.value;
        wrapper.dataset.ccDefaultCaptured = "1";

        function check() {
            // Read from numberEl regardless of which input actually fired
            // the event -- Gradio's own reactive binding keeps both in
            // sync even though it only dispatches a native event on the
            // one the user directly interacted with.
            //
            // Compared as NUMBERS, not raw strings -- confirmed live that
            // typing a value back to default via the number field can
            // settle on a differently-formatted (but numerically equal)
            // string than a range-driven update does (e.g. "0" vs
            // "0.00"), which a strict string comparison would never
            // recognize as "back to default".
            wrapper.classList.toggle('colorcraft-changed', parseFloat(numberEl.value) !== parseFloat(wrapper.dataset.ccDefault));
        }
        [numberEl, rangeEl].filter(Boolean).forEach((el) => {
            el.addEventListener('input', check);
            el.addEventListener('change', check);
        });
        wiredChecks.set(wrapper, check);
    }

    function watchDropdown(wrapper) {
        if (wrapper.dataset.ccDefaultCaptured) return;
        const inputEl = wrapper.querySelector('input');
        if (!inputEl) return;
        wrapper.dataset.ccDefault = inputEl.value;
        wrapper.dataset.ccDefaultCaptured = "1";

        function check() {
            wrapper.classList.toggle('colorcraft-changed', inputEl.value !== wrapper.dataset.ccDefault);
        }
        inputEl.addEventListener('input', check);
        inputEl.addEventListener('change', check);
        wiredChecks.set(wrapper, check);
    }

    function setupChangeHighlighting() {
        document.querySelectorAll('[id$="_colorcraft_accordion"]').forEach((accordion) => {
            accordion.querySelectorAll('.gradio-slider').forEach(watchSlider);
            accordion.querySelectorAll('.colorcraft-dropdown').forEach(watchDropdown);
        });
        // Re-run every already-known check too, not just newly-discovered
        // wrappers -- this is the actual fix for infotext restore, which
        // updates values without reliably firing input/change.
        wiredChecks.forEach((check) => check());
    }

    document.addEventListener('click', () => {
        wiredChecks.forEach((check) => check());
    });

    onUiLoaded(setupChangeHighlighting);
    onUiUpdate(setupChangeHighlighting);
})();