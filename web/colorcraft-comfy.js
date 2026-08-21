import { app } from "../../scripts/app.js";

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Custom link/socket colors, by type name. Edit this dict to add/change
// colors. Applied from nodeCreated rather than init()/setup(), since
// ComfyUI/LiteGraph resets these color maps after those run -- nodeCreated
// fires per node, after that reset, so it reliably sticks.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const LINK_TYPE_COLORS = {
    COLORCRAFT_MASK: "#e33d2d",
    COLORCRAFT_MODIFIER: "#a4cc1f",
    COLORCRAFT_SCHEDULE: "#00a496",
    COLORCRAFT_DEBUG: "#8059ff",
};

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

const AXIS_WIDGET_COLOR_KEYS = {
    temperature: "temperature",
    tint: "tint",
    lab_a: "lab-a",
    lab_b: "lab-b",
    temp_plus_tint: "temp+tint",
    temp_minus_tint: "temp-tint",
    lab_a_plus_b: "lab-a+b",
    lab_a_minus_b: "lab-a-b",
};

function axisGradientStops(colorLo, colorHi, alpha, withGrey = true) {
    const toAlpha = (c) => c.replace("hsl(", "hsla(").replace(")", `, ${alpha})`);
    if (!withGrey) {
        return [
            [0, toAlpha(colorLo)],
            [1, toAlpha(colorHi)],
        ];
    }
    return [
        [0, toAlpha(colorLo)],
        [0.5, `hsla(0, 0%, 80%, ${alpha})`],
        [1, toAlpha(colorHi)],
    ];
}

// Computes the color_shift preview swatch from a node's own red/green/blue/
// brightness/color_shift_amount widget values. Display approximation only
// -- the real anchor color comes from actually VAE-encoding a flat-color
// image server-side (build_color_latent in nodes.py).
function computeShiftSwatch(node) {
    const val = (name) => node.widgets?.find((w) => w.name === name)?.value ?? 0;
    const amount = val("color_shift_amount");
    const brightness = val("brightness");
    const invert = amount < 0;
    // sqrt-compresses each channel's -2..2 range into a -1..1 display offset
    // so red=1 already reads as strongly red, with red=2 still visibly
    // pushing further, rather than both looking identically maxed out.
    const toOffset = (channel) => {
        const x = val(channel) + brightness;
        const mag = Math.min(Math.abs(x) / 2, 1);
        return Math.sign(x) * Math.sqrt(mag);
    };
    const toChannel = (offset) => {
        const v = Math.round(Math.max(0, Math.min(1, 0.5 + 0.5 * offset)) * 255);
        // amount < 0 pushes away from this color rather than toward it --
        // showing the complementary color as a rough intuitive cue.
        return invert ? 255 - v : v;
    };
    const r = toChannel(toOffset("red"));
    const g = toChannel(toOffset("green"));
    const b = toChannel(toOffset("blue"));
    // Swatch shows WHICH color, not exact intensity -- amount=1 already
    // reads as a strong shift, so it's clamped to +-1 before sqrt rather
    // than scaled against the slider's full +-10 range.
    const alpha = Math.sqrt(Math.min(Math.abs(amount), 1));
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function applyLinkTypeColors() {
    if (app.canvas) {
        if (!app.canvas.default_connection_color_byType) {
            app.canvas.default_connection_color_byType = {};
        }
        Object.assign(app.canvas.default_connection_color_byType, LINK_TYPE_COLORS);
    }
    if (window.LGraphCanvas) {
        Object.assign(window.LGraphCanvas.link_type_colors, LINK_TYPE_COLORS);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Shared layout primitives (spacers, plots, boxes, accordions) -- generic
// across every Colorcraft node type. Per-node BOXES/SPACERS_BEFORE/
// ACCORDIONS live in NODE_CONFIGS at the bottom.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const SPACER_HEIGHT = 6;
const LABEL_HEIGHT = 20;
const LABEL_OFFSET = 4;
const LABEL_INDENT = 14;
const ACCORDION_LABEL_INDENT = 16;
const ACCORDION_HEIGHT = 14;

function spacerName(before) {
    return `__spacer_before_${before}`;
}

function makeSpacer({ before, height = SPACER_HEIGHT, label = null } = {}) {
    return {
        type: "colorcraft_spacer",
        name: spacerName(before),
        value: null,
        options: { serialize: false },
        computeSize(width) {
            return [width, label ? LABEL_HEIGHT : height];
        },
        draw(ctx, node, widgetWidth, y, widgetHeight) {
            if (!label) return;
            ctx.save();
            ctx.fillStyle = "rgba(255,255,255,0.5)";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "bottom";
            const text = label.toUpperCase();
            ctx.fillText(text, LABEL_INDENT, y + LABEL_HEIGHT - LABEL_OFFSET);
            const textWidth = ctx.measureText(text).width;       
            ctx.strokeStyle = "rgba(255,255,255,0.5)";       
            ctx.beginPath();     
            ctx.moveTo(LABEL_INDENT + 4 + textWidth, y + widgetHeight / 2);        
            ctx.lineTo(widgetWidth - LABEL_INDENT - 4, y + widgetHeight / 2);      
            ctx.stroke();
            ctx.restore();
        },
        mouse() {
            return false;
        },
        serializeValue() {
            return undefined;
        },
    };
}

function insertBefore(node, targetName, widget) {
    const idx = node.widgets.findIndex((w) => w.name === targetName);
    if (idx === -1) node.widgets.push(widget);
    else node.widgets.splice(idx, 0, widget);
}

function insertAfter(node, targetName, widget) {
    const idx = node.widgets.findIndex((w) => w.name === targetName);
    if (idx === -1) node.widgets.push(widget);
    else node.widgets.splice(idx + 1, 0, widget);
}

function widgetBottom(node, widget) {
    const height = widget.computeSize
        ? widget.computeSize(node.size[0])[1]
        : LiteGraph.NODE_WIDGET_HEIGHT || 20;
    return widget.last_y + height;
}

function val(node, name, fallback) {
    const w = node.widgets?.find((w) => w.name === name);
    return w ? w.value : fallback;
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Schedule preview plot
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const PLOT_HEIGHT = 40;
const PLOT_MARGIN = 18;
const PLOT_PADDING = 4;
const PLOT_RESOLUTION = 200;

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

const PLOT_STEPS_DEFAULT = 8;

// plot_steps's widget default (8) doubles as the "untouched" signal for the
// global fallback -- only affects the preview plot's tick marks, never
// actual render output. Falls back to the widget's own value if the
// settings API isn't available.
function getEffectivePlotSteps(node) {
    const widgetVal = val(node, "plot_steps", PLOT_STEPS_DEFAULT);
    if (widgetVal !== PLOT_STEPS_DEFAULT) return widgetVal;
    try {
        const settingVal = app.extensionManager?.setting?.get?.("Colorcraft.DefaultPlotSteps");
        if (typeof settingVal === "number" && !Number.isNaN(settingVal)) return settingVal;
    } catch {
        // Settings API unavailable -- fall through to the widget default.
    }
    return widgetVal;
}

function getScheduleParams(node) {
    return {
        start: val(node, "start", 0),
        end: val(node, "end", 1),
        bias: val(node, "bias", 0.5),
        amount: val(node, "strength", 1),
        exponent: val(node, "advanced", false) ? val(node, "exponent", 1) : 0,
        start_off: val(node, "advanced", false) ? val(node, "start_off", 0) : 0,
        end_off: val(node, "advanced", false) ? val(node, "end_off", 0) : 0,
        smooth: val(node, "smooth", true),
        plot_steps: getEffectivePlotSteps(node),
    };
}

function makePlotWidget() {
    return {
        type: "colorcraft_plot",
        name: "__schedule_plot",
        value: null,
        options: { serialize: false },
        computeSize(width) {
            return [width, PLOT_HEIGHT + PLOT_PADDING * 2];
        },
        draw(ctx, node, widgetWidth, y, widgetHeight) {
            const tickH = 3;
            const labelH = 10;
            const plotY = y;
            const x0 = PLOT_MARGIN;
            const x1 = widgetWidth - PLOT_MARGIN;
            const plotW = x1 - x0;
            const plotH = PLOT_HEIGHT;
            const toX = (t) => x0 + t * plotW;
            const toY = (v) => plotY + plotH * (1 - (v + 1) / 2);

            const p = getScheduleParams(node);
            const { arr, mid } = makeScheduleArray(PLOT_RESOLUTION, p);
            const stroke = "rgba(255,255,255,0.55)";
            const stepDivisor = Math.max(1, Math.round(p.plot_steps) - 1);

            ctx.save();
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;

            ctx.strokeRect(x0, plotY, plotW, plotH);

            // Bottom ticks and labels
            ctx.fillStyle = stroke;
            ctx.font = `${labelH - 1}px sans-serif`;
            ctx.textBaseline = 'top';
            ctx.textAlign = 'center';
            const labelEvery = stepDivisor > 39 ? 5 : stepDivisor > 19 ? 2 : 1;
            for (let i = 0; i <= stepDivisor; i++) {
                const xx = toX(i / stepDivisor);
                ctx.beginPath(); ctx.moveTo(xx, plotY + plotH); ctx.lineTo(xx, plotY + plotH + tickH); ctx.stroke();
                if (i % labelEvery === 0) ctx.fillText(String(i), xx, plotY + plotH + tickH + PLOT_PADDING / 2);
            }

            ctx.setLineDash([1, 3]);
            ctx.beginPath(); ctx.moveTo(x0, toY(0)); ctx.lineTo(x1, toY(0)); ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 0.5;
            if (p.exponent != 0) {
                ctx.beginPath(); ctx.moveTo(toX(mid), plotY); ctx.lineTo(toX(mid), plotY + plotH); ctx.stroke();
            }
            ctx.beginPath(); ctx.moveTo(toX(p.start), plotY); ctx.lineTo(toX(p.start), plotY + plotH); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(toX(p.end), plotY); ctx.lineTo(toX(p.end), plotY + plotH); ctx.stroke();
            ctx.setLineDash([]);

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
        },
        mouse() {
            return false;
        },
        serializeValue() {
            return undefined;
        },
    };
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Mask shape preview plot
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const MASK_PLOT_HEIGHT = 90;
const MASK_PLOT_MARGIN = 18;
const MASK_PLOT_RESOLUTION = 200;
const MASK_PLOT_XMIN = -1.1;
const MASK_PLOT_XMAX = 1.1;
const MASK_PLOT_XTICKS = [-1, -0.5, 0, 0.5, 1];

// Hue is circular, so it gets its own +-pi domain instead of the shared
// linear-axis one -- HEADROOM is a small visual overshoot past +-pi so the
// curve/gradient are seen continuing a little past each edge rather than
// cutting off hard right at the wrap seam.
const HUE_PLOT_HEADROOM = 0.35;
const HUE_PLOT_XTICKS = [
    [-Math.PI, "-\u03c0"],
    [-Math.PI / 2, "-\u03c0/2"],
    [0, "0"],
    [Math.PI / 2, "\u03c0/2"],
    [Math.PI, "\u03c0"],
];

function getMaskParams(node) {
    return {
        axis: val(node, "mask_axis", "exposure"),
        mode: val(node, "mask_mode", "range"),
        center: val(node, "mask_center", 0),
        width: val(node, "mask_width", 0),
        hardness: val(node, "mask_hardness", 1),
        strength: val(node, "mask_strength", 1),
    };
}

// Mirrors HARDNESS_GAIN/_mask_shape on the Python side exactly -- this is
// a preview, so it has to compute the same thing the real mask does.
const HARDNESS_GAIN = 5.0;

function maskShape(mode, hardness, width, strength, diff) {
    // `diff` is already center-relative (x - center) -- shared by both the
    // linear axes and the circular hue axis (which wraps `diff` first).
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
    // Wraps into (-pi, pi] -- same formula as compute_hue_mask's Python:
    // (angle - center + pi) % (2*pi) - pi, just without the center term
    // folded in (callers subtract center before or after wrapping as needed).
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
    for (const [t, color] of stops) {
        gradient.addColorStop(t, color);
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(toX(plotXmin), toY(0) + 2, toX(plotXmax) - toX(plotXmin), 8);
    ctx.restore();
}

function makeMaskPlotWidget() {
    return {
        type: "colorcraft_mask_plot",
        name: "__mask_plot",
        value: null,
        options: { serialize: false },
        computeSize(width) {
            return [width, MASK_PLOT_HEIGHT + SPACER_HEIGHT * 2];
        },
        draw(ctx, node, widgetWidth, y, widgetHeight) {
            const p = getMaskParams(node);
            const isHue = p.axis === "hue";
            // Small extra domain past +-pi so the curve/gradient visibly
            // continue a little past each edge instead of cutting off hard
            // right at the wrap point -- purely a legibility aid, the actual
            // wrap math (hueMaskValue/wrapAngle) doesn't need it.
            const xmin = isHue ? -Math.PI - HUE_PLOT_HEADROOM : MASK_PLOT_XMIN;
            const xmax = isHue ? Math.PI + HUE_PLOT_HEADROOM : MASK_PLOT_XMAX;

            const x0 = MASK_PLOT_MARGIN;
            const x1 = widgetWidth - MASK_PLOT_MARGIN;
            const plotW = x1 - x0;
            const plotH = MASK_PLOT_HEIGHT;
            const toX = (v) => x0 + ((v - xmin) / (xmax - xmin)) * plotW;
            const toY = (v) => y + plotH * (1 - (v + 1.1) / 2.2);

            const stroke = "rgba(255,255,255,0.55)";

            if (p.axis === "exposure") {
                // Now that raw values are normalized by exposure_scale before
                // reaching mask_center, +-1 is the calibrated black/white
                // transition on EITHER model -- no more per-model number
                // needed here. Beyond +-1 the gradient holds solid black/
                // white out to the plot's own +-1.1 edge (canvas gradients
                // clamp to the boundary stop color past the defined range).
                const alpha = 0.75;
                drawGradientStrip(ctx, toX, toY, -1, 1, xmin, xmax, [
                    [0, `hsla(0, 0%, 0%, ${alpha})`],
                    [1, `hsla(0, 0%, 100%, ${alpha})`],
                ]);
            }

            if (isHue) {
                // The 220-degree rotation offset was eyeballed against Krea2's hue geometry.
                // hueAt wraps its input first, so headroom positions past +-pi still resolve
                // to the correct (repeated) color instead of an out-of-range hue.
                const alpha = 0.5;
                const hueAt = (x) => {
                    const angle = wrapAngle(x);
                    const raw = 220 - ((angle + Math.PI) / (2 * Math.PI)) * 360;
                    return ((raw % 360) + 360) % 360;
                };
                const stepsPerPeriod = 24;
                const dx = (xmax - xmin) / stepsPerPeriod;
                const sampleXs = [];
                for (let x = xmin; x < xmax; x += dx) {
                    sampleXs.push(x);
                }
                sampleXs.push(xmax);
                const stops = sampleXs.map((x) => [
                    (x - xmin) / (xmax - xmin),
                    `hsla(${hueAt(x)}, 100%, 50%, ${alpha})`,
                ]);
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

            ctx.strokeRect(x0, y, plotW, plotH);

            ctx.font = "9px sans-serif";
            ctx.fillStyle = stroke;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const ticks = isHue ? HUE_PLOT_XTICKS : MASK_PLOT_XTICKS.map((v) => [v, String(v)]);
            for (const [v, label] of ticks) {
                const xx = toX(v);
                ctx.beginPath();
                ctx.moveTo(xx, y + plotH);
                ctx.lineTo(xx, y + plotH + 3);
                ctx.stroke();
                ctx.fillText(label, xx, y + plotH + 4);
            }

            if (!isHue) {
                // Rough backdrop cue for where real values typically cluster
                // on a +-1-normalized linear axis. Skipped for hue entirely --
                // there's no equivalent "typical clustering" shape for a
                // circular axis, so drawing one here would just be decorative
                // noise with no real meaning, not a simplified truth.
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
            // position on the plot -- unlike a linear axis, where a center
            // far outside the domain is legitimately "off-plot".
            const centerPos = isHue ? wrapAngle(p.center * Math.PI) : p.center;
            if (centerPos >= xmin && centerPos <= xmax) {
                ctx.save();
                ctx.setLineDash([5.5, 5]);
                ctx.beginPath();
                const centerX = toX(centerPos);
                ctx.moveTo(centerX, y);
                ctx.lineTo(centerX, y + plotH);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }

            ctx.save();
            ctx.beginPath();
            ctx.rect(x0, y, plotW, plotH);
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
                if (i === 0) ctx.moveTo(xx, yy);
                else ctx.lineTo(xx, yy);
            }
            ctx.stroke();
            ctx.restore();

            ctx.restore();
        },
        mouse() {
            return false;
        },
        serializeValue() {
            return undefined;
        },
    };
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Generic accordion mechanism
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

function resyncInputLinkSlots(node) {
    if (!node.inputs || !node.graph) return;
    for (let i = 0; i < node.inputs.length; i++) {
        const link = node.getInputLink?.(i);
        if (link) link.target_slot = i;
    }
}

function hideWidget(widget, node) {
    if (widget.hidden) return;
    widget.hidden = true;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.type = "colorcraft_hidden";
    widget.computeSize = () => [0, -4];

    if (node?.inputs) {
        const idx = node.inputs.findIndex((i) => i.name === widget.name);
        if (idx !== -1) {
            if (node.inputs[idx].link != null) node.disconnectInput(idx);
            widget._savedInput = node.inputs[idx];
            widget._savedInputIndex = idx;
            node.inputs.splice(idx, 1);
            resyncInputLinkSlots(node);
        }
    }
}

function showWidget(widget, node) {
    if (!widget.hidden) return;
    widget.hidden = false;
    widget.type = widget.origType;
    widget.computeSize = widget.origComputeSize;
    delete widget.origType;
    delete widget.origComputeSize;

    if (node?.inputs && widget._savedInput && !node.inputs.some((i) => i.name === widget.name)) {
        const idx = Math.min(widget._savedInputIndex, node.inputs.length);
        node.inputs.splice(idx, 0, widget._savedInput);
        resyncInputLinkSlots(node);
    }
    delete widget._savedInput;
    delete widget._savedInputIndex;
}

function setGroupVisibility(node, showNames, hideNames, { resize = true } = {}) {
    const prevFit = resize ? node.computeSize() : null;

    for (const name of showNames) {
        const w = node.widgets.find((widget) => widget.name === name);
        if (w) showWidget(w, node);
    }
    for (const name of hideNames) {
        const w = node.widgets.find((widget) => widget.name === name);
        if (w) hideWidget(w, node);
    }

    if (resize) {
        const newFit = node.computeSize();
        const deltaHeight = newFit[1] - prevFit[1];
        const width = Math.max(node.size[0], newFit[0]);
        const height = Math.max(node.size[1] + deltaHeight, newFit[1]);
        node.setSize([width, height]);
    }

    node.graph?.setDirtyCanvas(true, true);
    app.canvas?.draw(true, true);
}

// Recomputes the hidden-set from scratch off every accordion's current toggle
// value plus any conditional rules (e.g. a widget only relevant for certain
// combo values), rather than one incremental show/hide per event.
function refreshVisibility(node, config, opts) {
    const hidden = new Set();
    for (const { toggle, body } of config.accordions) {
        if (!val(node, toggle, false)) {
            for (const name of body) hidden.add(name);
        }
    }
    for (const { widget, dependsOn, show } of config.conditionals ?? []) {
        if (!show(val(node, dependsOn))) hidden.add(widget);
    }

    const show = [];
    const hide = [];
    for (const w of node.widgets) {
        (hidden.has(w.name) ? hide : show).push(w.name);
    }
    setGroupVisibility(node, show, hide, opts);
}

function makeAccordionHeader(node, { toggle, label }, config) {
    const w = node.widgets.find((widget) => widget.name === toggle);
    if (!w) return null;

    w.computeSize = () => [0, ACCORDION_HEIGHT];
    w.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        ctx.save();
        ctx.fillStyle = "rgba(255,255,255,0.5)";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const arrow = w.value ? "\u25BC" : "\u25B6";
        ctx.fillText(arrow, ACCORDION_LABEL_INDENT, y + ACCORDION_HEIGHT / 2 + 0.75);
        ctx.fillText(label, ACCORDION_LABEL_INDENT + 10, y + ACCORDION_HEIGHT / 2 + 1);
        ctx.restore();
    };

    const origCallback = w.callback;
    w.callback = function (value, ...rest) {
        origCallback?.call(this, value, ...rest);
        refreshVisibility(node, config);
        // Boxes (drawn from widgets' own last_y in onDrawForeground) were
        // reading stale positions until the next unrelated redraw (e.g. a
        // mouse move) -- same one-frame-later timing quirk as onNodeCreated's
        // comment above; re-applying after a frame fixes it here too.
        requestAnimationFrame(() => refreshVisibility(node, config));
    };

    return w;
}

// Hooks each conditional rule's `dependsOn` widget so changing it re-runs
// refreshVisibility -- e.g. mask_width only makes sense for range/protect
// range, so it needs to react to mask_mode changing, not just accordion
// toggles.
function hookConditionalTriggers(node, config) {
    const hooked = new Set();
    for (const { dependsOn } of config.conditionals ?? []) {
        if (hooked.has(dependsOn)) continue;
        hooked.add(dependsOn);
        const w = node.widgets.find((widget) => widget.name === dependsOn);
        if (!w) continue;
        const origCallback = w.callback;
        w.callback = function (value, ...rest) {
            origCallback?.call(this, value, ...rest);
            refreshVisibility(node, config);
            requestAnimationFrame(() => refreshVisibility(node, config));
        };
    }
}

// Forces a canvas redraw for widgets that change what the mask plot draws
// but don't drive any show/hide rules -- e.g. selecting a new mask_axis
// from a combo popup doesn't naturally trigger a repaint the way dragging
// a slider does.
function hookMaskPlotRedraw(node) {
    for (const name of ["mask_axis"]) {
        const w = node.widgets.find((widget) => widget.name === name);
        if (!w) continue;
        const origCallback = w.callback;
        w.callback = function (value, ...rest) {
            origCallback?.call(this, value, ...rest);
            node.graph?.setDirtyCanvas(true, true);
            app.canvas?.draw(true, true);
            requestAnimationFrame(() => {
                node.graph?.setDirtyCanvas(true, true);
                app.canvas?.draw(true, true);
            });
        };
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Per-node-type layout config.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const NODE_CONFIGS = {
    ColorcraftBasic: {
        boxes: [
            ["strength", "start", "end", "advanced", "bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"],
            ["contrast"],
            ["color_shift_amount", "mode", "red", "green", "blue", "brightness"],
        ],
        spacers: [
            { before: "strength" },
            { before: "__schedule_plot" },
            { before: "contrast", label: "punch" },
            { before: "color_shift_amount", label: "color shift" },
        ],
        accordions: [
            { toggle: "advanced", label: "ADVANCED SCHEDULE", body: ["bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"] },
        ],
        hasMaskPlot: false,
    },
    ColorcraftAdvanced: {
        boxes: [
            ["strength", "start", "end", "advanced", "bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"],
            ["exposure", "tone_compression"],            
            ["contrast", "clarity", "sharpness"],
            ["temperature", "tint"],
            ["more_colors", "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b"],
            ["vibrance", "saturation"],
            ["chroma_contrast", "chroma_center"],
            ["color_shift", "color_shift_amount", "mode", "red", "green", "blue", "brightness"],
            ["dev", "recenter_override", "recenter", "max_chroma_override", "max_chroma", "chroma_plane_override", "chroma_plane"],
        ],
        spacers: [
            { before: "strength" },
            { before: "__schedule_plot" },
            { before: "exposure", label: "luma" },            
            { before: "contrast", label: "punch" },
            { before: "temperature", label: "chroma" },            
            { before: "vibrance" },
            { before: "chroma_contrast" },
            { before: "more_colors" },
            { before: "color_shift", label: "shift" },
            { before: "dev", label: "hic svnt dracones" },
        ],
        accordions: [
            { toggle: "advanced", label: "ADVANCED SCHEDULE", body: ["bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"] },
            { toggle: "more_colors", label: "MORE COLORS", body: ["temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b"] },
            { toggle: "color_shift", label: "COLOR SHIFT", body: ["color_shift_amount", "mode", "red", "green", "blue", "brightness"] },
            { toggle: "dev", label: "DEV", body: ["recenter_override", "recenter", "max_chroma_override", "max_chroma", "chroma_plane_override", "chroma_plane"] },
        ],
        hasMaskPlot: false,
    },
    ColorcraftSchedule: {
        boxes: [
            ["strength", "start", "end", "advanced", "bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"],
        ],
        spacers: [
            { before: "strength" },
            { before: "__schedule_plot" },
        ],
        accordions: [
            { toggle: "advanced", label: "ADVANCED SCHEDULE", body: ["bias", "exponent", "start_off", "end_off", "smooth", "plot_steps"] },
        ],
        hasMaskPlot: false,
    },
    ColorcraftLuma: {
        boxes: [
            ["exposure", "tone_compression"],
        ],
        spacers: [
            { before: "exposure" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftChroma: {
        boxes: [
            ["temperature", "tint"],
            ["vibrance", "saturation"],
            ["chroma_contrast", "chroma_center"],
        ],
        spacers: [
            { before: "temperature" },
            { before: "vibrance" },
            { before: "chroma_contrast" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftChromaPlus: {
        boxes: [
            ["temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b"],
        ],
        spacers: [
            { before: "temp_plus_tint" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftPunch: {
        boxes: [
            ["contrast", "clarity", "sharpness"],
        ],
        spacers: [
            { before: "contrast" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftShift: {
        boxes: [
            ["color_shift_amount", "mode", "red", "green", "blue", "brightness"],
        ],
        spacers: [
            { before: "color_shift_amount" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftMaskPreview: {
        boxes: [["color"]],
        spacers: [
            { before: "color" },
        ],
        accordions: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftMasking: {
        boxes: [["mask_mode", "mask_axis", "mask_strength", "mask_width", "mask_center", "mask_hardness"]],
        spacers: [
            { before: "mask_mode" },
            { before: "__mask_plot" },
        ],
        accordions: [],
        conditionals: [
            { widget: "mask_width", dependsOn: "mask_mode", show: (v) => v === "range" || v === "protect range" },
            { widget: "mask_strength", dependsOn: "mask_mode", show: (v) => v !== "split" },
        ],
        hasMaskPlot: true,
        hasSchedule: false,
    },
    ColorcraftMaskCombine: {
        boxes: [["operation"]],
        spacers: [
            { before: "operation" },
        ],
        accordions: [],
        conditionals: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
    ColorcraftMaskBlur: {
        boxes: [["radius", "spread"]],
        spacers: [
            { before: "radius" },
        ],
        accordions: [],
        conditionals: [],
        hasMaskPlot: false,
        hasSchedule: false,
    },
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Custom Numeric Widget
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

const DEFAULT_VALUE_COLOR = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
const CHANGED_VALUE_COLOR = "#ddd";
const ARROW_ACTIVE_COLOR = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
const ARROW_DISABLED_COLOR = "#666";
const ARROW_W = 10;
const ARROW_H = 10;
const ARROW_INSET = 11; 
const TEXT_INDENT_LEFT = 5;
const TEXT_INDENT_RIGHT = 20;

function drawArrow(ctx, cx, cy, direction, enabled) {
  ctx.fillStyle = enabled ? ARROW_ACTIVE_COLOR : ARROW_DISABLED_COLOR;
  ctx.beginPath();
  if (direction === "left") {
    ctx.moveTo(cx + ARROW_W / 2, cy - ARROW_H / 2);
    ctx.lineTo(cx - ARROW_W / 2, cy);
    ctx.lineTo(cx + ARROW_W / 2, cy + ARROW_H / 2);
  } else {
    ctx.moveTo(cx - ARROW_W / 2, cy - ARROW_H / 2);
    ctx.lineTo(cx + ARROW_W / 2, cy);
    ctx.lineTo(cx - ARROW_W / 2, cy + ARROW_H / 2);
  }
  ctx.closePath();
  ctx.fill();
}

function getBoundState(widget) {
  const { min, max } = widget.options || {};
  const atMin = min != null && widget.value <= min;
  const atMax = max != null && widget.value >= max;
  return [atMin, atMax];
}

// Whether the mouse is currently over a widget's own row, in node-local
// coordinates. `app.canvas.graph_mouse` is already in graph space (pan/zoom
// applied), so this only needs to subtract the node's own position -- no
// extra invalidation needed either, since LiteGraph's own mousemove handler
// already marks the canvas dirty and redraws continuously while hovering.
function isWidgetHovered(node, y, H, margin, widgetWidth) {
    const mouse = app.canvas?.graph_mouse;
    if (!mouse || !node.pos) return false;
    const localX = mouse[0] - node.pos[0];
    const localY = mouse[1] - node.pos[1];
    return localX >= margin && localX <= widgetWidth - margin && localY >= y && localY <= y + H;
}

function makeNumericWidgetDraw(widget, defaultValue, axisColors, isShiftAmount) {
    return function (ctx, node, widgetWidth, y) {
        const H = LiteGraph.NODE_WIDGET_HEIGHT || 20;
        const margin = 15;
        const midY = y + H * 0.5;

        ctx.save();

        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#666";
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(margin, y, widgetWidth - margin * 2, H, H * 0.5);
        } else {
            ctx.rect(margin, y, widgetWidth - margin * 2, H);
        }
        ctx.fill();
        ctx.stroke();

        if (axisColors && isWidgetHovered(node, y, H, margin, widgetWidth)) {
            // Thin gradient strip along the pill's bottom edge showing which
            // direction this slider pushes color. Clipped to the pill's own
            // roundRect path when available so it follows the fillet. Hover-
            // only -- eight of these always-on was too much visual noise.
            const [colorLo, colorHi] = axisColors;
            const stripH = 4;
            const stripY = y + H - stripH;
            ctx.save();
            ctx.beginPath();
            let clipped = false;
            if (ctx.roundRect) {
                ctx.roundRect(margin, y, widgetWidth - margin * 2, H, H * 0.5);
                ctx.clip();
                clipped = true;
            }
            const stripX0 = clipped ? margin : margin + H * 0.5;
            const stripX1 = clipped ? widgetWidth - margin : widgetWidth - margin - H * 0.5;
            const gradient = ctx.createLinearGradient(stripX0, 0, stripX1, 0);
            for (const [t, color] of axisGradientStops(colorLo, colorHi, 0.6)) {
                gradient.addColorStop(t, color);
            }
            ctx.fillStyle = gradient;
            ctx.fillRect(stripX0, stripY, stripX1 - stripX0, stripH);
            ctx.restore();
        } else if (isShiftAmount) {
            // Flat-color swatch (not a gradient -- one resolved color, not a
            // range) along color_shift_amount's own pill, tied to the SAME
            // red/green/blue/brightness values color_shift will actually use.
            // Always on (not hover-gated like the axis gradients above) since
            // its own alpha already IS color_shift_amount, scaled -- it's
            // naturally invisible at amount=0 rather than needing a hover
            // gate to hide it.
            const stripH = 4;
            const stripY = y + H - stripH;
            ctx.save();
            ctx.beginPath();
            let clipped = false;
            if (ctx.roundRect) {
                ctx.roundRect(margin, y, widgetWidth - margin * 2, H, H * 0.5);
                ctx.clip();
                clipped = true;
            }
            const stripX0 = clipped ? margin : margin + H * 0.5;
            const stripX1 = clipped ? widgetWidth - margin : widgetWidth - margin - H * 0.5;
            ctx.fillStyle = computeShiftSwatch(node);
            ctx.fillRect(stripX0, stripY, stripX1 - stripX0, stripH);
            ctx.restore();
        }

        const [atMin, atMax] = getBoundState(widget);
        drawArrow(ctx, margin + ARROW_INSET, midY, "left", !atMin);
        drawArrow(ctx, widgetWidth - margin - ARROW_INSET, midY, "right", !atMax);

        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        const label = widget.label ?? widget.name;
        ctx.fillText(label, margin * 2 + TEXT_INDENT_LEFT, midY);

        let displayValue = widget.value;
        if (widget.type === "number" && typeof displayValue === "number") {
            const precision = widget.options?.precision;
            displayValue = precision != null ? displayValue.toFixed(precision) : displayValue;
        }
        const isDefault = widget.value === defaultValue;
        ctx.fillStyle = isDefault ? DEFAULT_VALUE_COLOR : CHANGED_VALUE_COLOR;
        ctx.textAlign = "right";
        ctx.fillText(
            String(displayValue),
            widgetWidth - margin * 2 - TEXT_INDENT_RIGHT,
            midY
        );

        ctx.restore();
    };
}

function hookNumericWidgetDraw(node) {
    if (!node.widgets) return;
    for (const w of node.widgets) {
        if (w.type !== "number" && w.type !== "slider" && w.type !== "combo") continue;
        // widget.options.default isn't reliably populated -- fall back to
        // the value at hook time (i.e. at node creation, matching a freshly
        // placed node's real default) when it's missing.
        const defaultValue = w.options?.default ?? w.value;
        const axisKey = AXIS_WIDGET_COLOR_KEYS[w.name];
        const axisColors = axisKey ? LINEAR_AXIS_COLORS[axisKey] : null;
        const isShiftAmount = w.name === "color_shift_amount";
        w.draw = makeNumericWidgetDraw(w, defaultValue, axisColors, isShiftAmount);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Register Extension
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

app.registerExtension({
    name: "Colorcraft.ModifierLayout",
    settings: [
        {
            id: "Colorcraft.DefaultPlotSteps",
            name: "Default plot steps",
            category: ["Colorcraft", "Schedule Plot", "Default plot steps"],
            type: "slider",
            attrs: { min: 2, max: 50, step: 1 },
            defaultValue: 8,
        },
    ],
    nodeCreated(node) {
        applyLinkTypeColors();
        if (!NODE_CONFIGS[node.comfyClass]) return;
        const originalComputeSize = node.computeSize;
        node.computeSize = function(out) {
            let size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [250, 250];
            size[0] = Math.max(size[0], 250); 
            return size;
        };
        hookNumericWidgetDraw(node);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData.name];
        if (!config) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            if (config.hasSchedule !== false) insertAfter(this, "plot_steps", makePlotWidget());
            if (config.hasMaskPlot) insertBefore(this, "mask_mode", makeMaskPlotWidget());
            for (const { before, label } of config.spacers) {
                insertBefore(this, before, makeSpacer({ before, label }));
            }

            for (const accordion of config.accordions) makeAccordionHeader(this, accordion, config);
            hookConditionalTriggers(this, config);
            if (config.hasMaskPlot) hookMaskPlotRedraw(this);

            this.setSize(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);

            refreshVisibility(this, config);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            refreshVisibility(this, config, { resize: false });
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed || !this.widgets?.length) return;

            const findWidget = (name) => this.widgets.find((w) => w.name === name);

            ctx.save();
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.fillStyle = "rgba(255,255,255,0.25)";
            for (const names of config.boxes) {
                const ws = names.map(findWidget).filter((w) => w && w.last_y != null && !w.hidden);
                if (ws.length < 1) continue;
                const top = Math.min(...ws.map((w) => w.last_y)) - 3.5;
                const bottom = Math.max(...ws.map((w) => widgetBottom(this, w))) + 3.5;
                ctx.beginPath();
                ctx.roundRect(12, top, this.size[0] - 24, bottom - top, 11);
                ctx.fill();
                ctx.stroke();
            }
            ctx.restore();
        };
    },
});