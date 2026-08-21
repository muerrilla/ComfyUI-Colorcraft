import os
import re
import sys

import gradio as gr
import torch
import numpy as np
from PIL import Image

import modules.scripts as scripts
from modules.script_callbacks import (
    on_cfg_denoiser, on_cfg_after_cfg, remove_callbacks_for_function, on_infotext_pasted,
)
from modules.shared import device
import modules.devices as devices
from modules.ui_components import InputAccordion

COLORCRAFT_ROOT = scripts.basedir()
if COLORCRAFT_ROOT not in sys.path:
    sys.path.insert(0, COLORCRAFT_ROOT)

from lib_colorcraft.schedule import make_schedule
from lib_colorcraft.vectors import apply_vector_offset, apply_vibrance, apply_chroma_contrast, apply_tone_compression, chroma_axes
from lib_colorcraft.color import apply_contrast, apply_color_shift
from lib_colorcraft.masking import MASK_AXIS_OPTIONS, resolve_mask_tensor, apply_mask_gate, compute_hue_projection, compute_saturation_projection
from lib_colorcraft.debug import (
    DEBUG_COMPOSITE_COLORS, DEBUG_OVERLAY_COLORS, DEBUG_AXIS_STYLES, DEBUG_COLORMAP_LUT, DEBUG_HUE_LUT,
    compute_axis_projection, downscale_latent_for_storage,
    annotate_debug_image, debug_tensor_to_images, composite_mask_images, render_hue_images,
    collect_leaves, build_curve_infos,
)
from lib_colorcraft.basis import (
    BASIS_FAMILIES, VAE_DOWNSCALE_FACTOR, MODEL_DEV_DEFAULTS, LATENT_FORMAT_TO_FAMILY,
    load_basis, resolve_dev,
)

VECTORS_DIR = os.path.join(COLORCRAFT_ROOT, "vectors")

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
MODIFIER_COUNT = 10
MASK_COUNT = 10
COMBO_COUNT = 5
MASK_MODE_OPTIONS = ["highs", "lows", "split", "range", "protect range"]
COMBO_OP_OPTIONS = ["and", "or", "subtract", "xor"]
COMBO_OP_SYMBOLS = {"and": "\u2227", "or": "\u2228", "subtract": "\u2212", "xor": "\u2295"}  # ∧ ∨ − ⊕
COLOR_SHIFT_MODE_OPTIONS = ["default", "legacy"]
# Real stored basis vectors, plus saturation -- hue (a circular angle)
# doesn't fit the same greyscale/colormap-of-scalar treatment every other
# entry gets, so it's handled via two special-cased entries instead (see
# render_hue_images): "hue" renders the raw angle as a hue-wheel color,
# unmodulated by chroma magnitude -- the literal thing the mask gate
# uses. "hue (weighted)" is the same angle, desaturated toward grey in
# proportion to chroma magnitude -- closer to what a human would call
# "the hue" of a near-neutral pixel, but not what the mask gate sees.
DEBUG_AXIS_OPTIONS = [
    "exposure", "temperature", "tint", "temp+tint", "temp-tint",
    "lab-a", "lab-b", "lab-a+b", "lab-a-b", "clarity", "sharpness", "saturation",
    "hue", "hue (weighted)",
]


# Canonical field lists -- build BOTH the UI component order and the
# process()/infotext parsing order from the same source, so the two can
# never silently drift apart.
MODIFIER_FIELDS = [
    "active", "mask", "strength", "start", "end", "advanced", "exponent", "bias", "start_off", "end_off", "smooth",
    "exposure", "tone_compression",
    "contrast", "clarity", "sharpness",
    "temperature", "tint",
    "vibrance", "saturation",
    "chroma_contrast", "chroma_center",
    "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b",
    "color_shift_amount", "color_shift_mode", "color_shift_red", "color_shift_green", "color_shift_blue", "color_shift_brightness",
]
MODIFIER_VALUE_FIELDS = [f for f in MODIFIER_FIELDS if f != "active"]  # 'active' is implied by tag presence, not written
LEAF_FIELDS = ["mask_axis", "mask_mode", "mask_strength", "mask_width", "mask_center", "mask_hardness", "blur", "spread", "contrast"]
COMBO_FIELDS = ["mask_a", "mask_b", "operation", "blur", "spread", "contrast"]

BOOL_FIELDS = {"advanced", "smooth"}
STR_FIELDS = {"color_shift_mode", "mask_axis", "mask_mode", "mask", "mask_a", "mask_b", "operation"}


def resolve_family(p):
    """p.sd_model.model_config.latent_format's class name resolves to the
    same values Comfy's latent_format does ("Wan21"/"Flux"), so this
    shares LATENT_FORMAT_TO_FAMILY with the Comfy side."""
    latent_format = getattr(getattr(p.sd_model, "model_config", None), "latent_format", None)
    if latent_format is None:
        return None
    return LATENT_FORMAT_TO_FAMILY.get(type(latent_format).__name__)


def build_color_latent_forge(p, red, green, blue, brightness):
    """Forge-specific (BCHW, p.sd_model.encode_first_stage/get_first_stage_
    encoding), unlike Comfy's vae.encode (BHWC). Uses a spatial-mean
    reduction, not a center-pixel crop."""
    img = torch.zeros(1, 3, 512, 512).to(device, dtype=devices.dtype_vae)
    img[:, 0, :, :] += red
    img[:, 1, :, :] += green
    img[:, 2, :, :] += blue
    img += brightness
    with torch.no_grad():
        latent = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(img))
    if latent.dim() == 5:
        latent = latent.squeeze(2)
    return latent.mean(dim=(2, 3))[0]


def decode_debug_latent(p, latent, width, height, is_5d=False):
    """Decodes a (possibly downscaled) latent snapshot back to a list of
    RGB PIL images, one per batch item, at the given output size -- the
    compositing base image when debug capture happens mid-generation.
    is_5d restores the frame dimension before decoding, mirroring the
    squeeze/unsqueeze this callback applies to x elsewhere."""
    latent = latent.to(device=device, dtype=devices.dtype_vae)
    if is_5d:
        latent = latent.unsqueeze(2)
    with torch.no_grad():
        img = p.sd_model.decode_first_stage(latent)
    if img.dim() == 5:
        img = img.squeeze(1)
    img = (img / 2 + 0.5).clamp(0.0, 1.0)
    images = []
    for b in range(img.shape[0]):
        arr = (img[b].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr, mode="RGB").resize((width, height), Image.BILINEAR)
        images.append(pil)
    return images


# ---------------------------------------------------------------------------
# Mask/combo topology
# ---------------------------------------------------------------------------

def compute_mask_activeness(mod_mask, combo_a, combo_b, modifier_active):
    # A combo can only ever reference an EARLIER combo (structural UI
    # rule), so completeness propagates correctly in a single forward
    # pass: if C2 references C1 and C1 is missing A or B, C2 can't
    # produce real output either, regardless of what selects it.
    combo_complete = [combo_a[i] != "none" and combo_b[i] != "none" for i in range(COMBO_COUNT)]
    for i in range(COMBO_COUNT):
        for ref in (combo_a[i], combo_b[i]):
            if ref.startswith("C") and not combo_complete[int(ref[1:]) - 1]:
                combo_complete[i] = False

    leaf_active = [False] * MASK_COUNT
    combo_active = [False] * COMBO_COUNT

    # Direct pass: each active modifier's own mask selection lights up
    # whatever it points at. Multiple modifiers selecting the SAME
    # mask/combo is fine here by construction -- no exclusivity to track.
    for k in range(MODIFIER_COUNT):
        if not modifier_active[k]:
            continue
        sel = mod_mask[k]
        if sel == "none":
            continue
        if sel.startswith("M"):
            leaf_active[int(sel[1:]) - 1] = True
        elif sel.startswith("C"):
            idx = int(sel[1:]) - 1
            if combo_complete[idx]:
                combo_active[idx] = True

    # Propagate combo activeness backward through combo->combo references
    # (same "combos can only reference earlier combos" rule as above).
    for j in range(COMBO_COUNT - 1, -1, -1):
        if not combo_active[j]:
            continue
        for ref in (combo_a[j], combo_b[j]):
            if ref.startswith("C"):
                combo_active[int(ref[1:]) - 1] = True

    # Propagate leaf activeness from active combos referencing them.
    for j in range(COMBO_COUNT):
        if not combo_active[j]:
            continue
        for ref in (combo_a[j], combo_b[j]):
            if ref.startswith("M") and not ref.startswith("C"):
                leaf_active[int(ref[1:]) - 1] = True

    return leaf_active, combo_active


def compute_modifier_targeted(mod_mask):
    """Which modifiers currently have a mask/combo selected at all --
    a direct, trivial function of a modifier's own "mask" field."""
    return [mod_mask[k] != "none" for k in range(MODIFIER_COUNT)]


def compute_mod_mask_choices(combo_a, combo_b):
    """Choices are identical for every modifier -- sharing is allowed, so
    there's no per-modifier exclusion. Only combos that are actually
    complete (both A and B set, transitively through earlier combo
    references) are offered."""
    combo_complete = [combo_a[i] != "none" and combo_b[i] != "none" for i in range(COMBO_COUNT)]
    for i in range(COMBO_COUNT):
        for ref in (combo_a[i], combo_b[i]):
            if ref.startswith("C") and not combo_complete[int(ref[1:]) - 1]:
                combo_complete[i] = False
    return ["none"] + [f"M{i+1}" for i in range(MASK_COUNT)] + [f"C{j+1}" for j in range(COMBO_COUNT) if combo_complete[j]]


def on_topology_change(*args):
    m, k = COMBO_COUNT, MODIFIER_COUNT
    mod_mask = list(args[0:k])
    combo_a = list(args[k:k + m])
    combo_b = list(args[k + m:k + 2 * m])
    modifier_active = list(args[k + 2 * m:2 * k + 2 * m])
    masking_enabled = args[2 * k + 2 * m]

    leaf_active, combo_active = compute_mask_activeness(mod_mask, combo_a, combo_b, modifier_active)
    mod_mask_choices = compute_mod_mask_choices(combo_a, combo_b)
    modifier_targeted = compute_modifier_targeted(mod_mask)

    if not masking_enabled:
        # Masking off is functionally identical to every mask selection
        # being "none" for generation -- tab highlighting reflects that
        # too. Choices stay computed normally regardless.
        leaf_active = [False] * MASK_COUNT
        combo_active = [False] * m
        modifier_targeted = [False] * k

    outputs = []
    outputs += [gr.update(value=v) for v in leaf_active]
    outputs += [gr.update(value=v) for v in combo_active]
    outputs += [gr.update(choices=mod_mask_choices) for _ in range(k)]
    outputs += [gr.update(value=v) for v in modifier_targeted]
    return outputs


def compute_combo_labels(combo_a, combo_b, combo_op):
    """Tab label text for each combo: 'C{n}' (its own tag) until both Mask A
    and B are set, then '{A} {op symbol} {B}' using the raw tag (M3, C1,
    etc.) -- no 'Apply to' shown here, the tab highlight already answers
    whether it's doing something, this answers what."""
    labels = []
    for i, (a, b, op) in enumerate(zip(combo_a, combo_b, combo_op)):
        if a == "none" or b == "none":
            labels.append(f"C{i+1}")
        else:
            labels.append(f"{a} {COMBO_OP_SYMBOLS.get(op, '?')} {b}")
    return labels


def on_combo_label_change(*args):
    m = COMBO_COUNT
    combo_a = list(args[0:m])
    combo_b = list(args[m:2 * m])
    combo_op = list(args[2 * m:3 * m])
    labels = compute_combo_labels(combo_a, combo_b, combo_op)
    return [gr.update(value=v) for v in labels]


# ---------------------------------------------------------------------------
# Mask spec construction
# ---------------------------------------------------------------------------

def build_leaf_spec(axis, mode, center, hardness, width, strength, blur, spread, contrast):
    leaf = {
        "mask_axis": axis, "mask_mode": mode, "mask_center": center,
        "mask_hardness": hardness, "mask_width": width, "mask_strength": strength,
    }
    if blur != 0 or spread != 0 or contrast != 0:
        return {"blur": blur, "spread": spread, "contrast": contrast, "a": leaf}
    return leaf


def build_all_mask_specs(leaf_args, combo_args, mod_mask_selections):
    """leaf_args: MASK_COUNT tuples in LEAF_FIELDS order. combo_args:
    COMBO_COUNT tuples in COMBO_FIELDS order. mod_mask_selections:
    MODIFIER_COUNT "none"/"M{n}"/"C{n}" values. Returns
    (modifier_mask_map, leaf_specs, combo_specs) -- leaf_specs/combo_specs
    are returned too so the Debug panel can look up any tag directly."""
    leaf_specs = {}
    for i, vals in enumerate(leaf_args):
        d = dict(zip(LEAF_FIELDS, vals))
        key = f"M{i+1}"
        leaf_specs[key] = build_leaf_spec(d["mask_axis"], d["mask_mode"], d["mask_center"],
                                           d["mask_hardness"], d["mask_width"], d["mask_strength"],
                                           d["blur"], d["spread"], d["contrast"])

    def resolve_ref(ref, combo_specs):
        if ref == "none":
            return None
        if ref.startswith("C"):
            return combo_specs.get(ref)
        return leaf_specs.get(ref)

    combo_specs = {}
    for i, vals in enumerate(combo_args):
        d = dict(zip(COMBO_FIELDS, vals))
        key = f"C{i+1}"
        a_spec = resolve_ref(d["mask_a"], combo_specs)
        b_spec = resolve_ref(d["mask_b"], combo_specs)
        if a_spec is None or b_spec is None:
            if d["mask_a"] != "none" or d["mask_b"] != "none":
                print(f"[Colorcraft] WARNING: Combo {key} has an unresolved Mask A/B reference "
                      f"(dangling or incomplete upstream combo) -- producing no output this run.")
            combo_specs[key] = None
            continue
        inner = {"operation": d["operation"], "a": a_spec, "b": b_spec}
        combo_specs[key] = {"blur": d["blur"], "spread": d["spread"], "contrast": d["contrast"], "a": inner} if (d["blur"] != 0 or d["spread"] != 0 or d["contrast"] != 0) else inner

    # Direct per-modifier lookup -- multiple modifiers selecting the same
    # mask/combo is fine, nothing to conflict over anymore.
    modifier_mask_map = {}
    for k, sel in enumerate(mod_mask_selections):
        if sel == "none":
            continue
        spec = leaf_specs.get(sel) if sel.startswith("M") else combo_specs.get(sel)
        if spec is not None:
            modifier_mask_map[ROMAN[k]] = spec

    return modifier_mask_map, leaf_specs, combo_specs


# ---------------------------------------------------------------------------
# Infotext -- tag-indexed (by roman numeral / M{n} / C{n}), not positional.
# With up to 10 modifier / 10 mask / 5 combo tabs, sparse non-contiguous
# usage is the expected case, so entries are keyed off each entry's own
# embedded tag rather than enumerate() position.
# ---------------------------------------------------------------------------

def _cast_field(field, raw):
    if field in BOOL_FIELDS:
        return raw == "True"
    if field in STR_FIELDS:
        return raw
    return float(raw)


def write_entries(tag_value_pairs, value_fields):
    parts = []
    for tag, values in tag_value_pairs:
        vals = ",".join(str(values[f]) for f in value_fields)
        parts.append(f"{tag}:{vals}")
    return ";".join(parts)


def parse_entries(raw, value_fields):
    result = {}
    if not raw:
        return result
    for entry in raw.split(";"):
        if ":" not in entry:
            continue
        tag, values_str = entry.split(":", 1)
        values = values_str.split(",")
        if len(values) > len(value_fields):
            # More values than current fields expect (e.g. a removed
            # field) -- fields only ever get appended at the end, never
            # inserted mid-list, so trimming the trailing extras is safe.
            values = values[:len(value_fields)]
        elif len(values) < len(value_fields):
            print(f"[Colorcraft] WARNING: infotext entry for '{tag}' has {len(values)} values, "
                  f"expected {len(value_fields)} -- skipping (likely a version mismatch).")
            continue
        result[tag] = {f: _cast_field(f, v) for f, v in zip(value_fields, values)}
    return result


def combine_infotext_sections(mod_str, mask_str, combo_str):
    """One 'Colorcraft' key instead of three -- each extra_generation_
    params entry costs its own 'Key: ' prefix in the rendered infotext.
    Labeled (not positional) so a genuinely empty section can't shift
    the ones after it."""
    return f"MOD({mod_str})MASK({mask_str})COMBO({combo_str})"


def extract_infotext_section(raw, label):
    m = re.search(rf"{re.escape(label)}\((.*?)\)", raw)
    return m.group(1) if m else ""


def parse_infotext(infotext, params):
    try:
        raw = params.get("Colorcraft", "")
        if not raw:
            return  # nothing to parse -- don't create a "Colorcraft" key at all, for anything to false-positive on later
        colorcraft = {}
        colorcraft["modifiers"] = parse_entries(extract_infotext_section(raw, "MOD"), MODIFIER_VALUE_FIELDS)
        for entry in colorcraft["modifiers"].values():
            entry["active_flag"] = True  # tag's mere presence in the string IS the active signal
        colorcraft["masks"] = parse_entries(extract_infotext_section(raw, "MASK"), LEAF_FIELDS)
        colorcraft["combos"] = parse_entries(extract_infotext_section(raw, "COMBO"), COMBO_FIELDS)
        params["Colorcraft"] = colorcraft
    except Exception as e:
        print(f"[Colorcraft] WARNING: failed to parse infotext: {e}")


on_infotext_pasted(parse_infotext)


def _extract(d, section, tag, field, default=None):
    return d.get("Colorcraft", {}).get(section, {}).get(tag, {}).get(field, default)


class Script(scripts.Script):

    def title(self):
        return "Colorcraft"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.infotext_fields = []
        all_components = []
        modifier_components, leaf_components, combo_components = [], [], []
        modifier_active_widgets = []
        modifier_targeted_checkboxes = []
        mod_mask_widgets = []
        leaf_active_checkboxes = []
        combo_a_widgets, combo_b_widgets, combo_active_checkboxes = [], [], []
        combo_op_widgets, combo_label_widgets = [], []

        with InputAccordion(False, label="Colorcraft", elem_id=self.elem_id("accordion")) as gr_enabled:
            all_components.append(gr_enabled)
            self.infotext_fields.append((gr_enabled, lambda d: "Colorcraft" in d))

            initial_mod_mask_choices = compute_mod_mask_choices(["none"] * COMBO_COUNT, ["none"] * COMBO_COUNT)
            with gr.Group(elem_classes=["colorcraft-tab-group", "colorcraft-mod-tab-group"]):
                for i in range(MODIFIER_COUNT):
                    tag = ROMAN[i]
                    with gr.Tab(tag, elem_classes=["colorcraft-tab", f"colorcraft-tab{i}"]):
                        w = {}
                        with gr.Row(elem_classes=["colorcraft-top-row"]):
                            w["active"] = gr.Checkbox(value=False, label="Active", elem_classes=["colorcraft-active"], min_width=60)
                            gr.HTML('<canvas class="colorcraft-schedule-plot"></canvas>') 
                            w["mask"] = gr.Dropdown(choices=initial_mod_mask_choices, value="none", label="Mask ➜", min_width=60, elem_classes=["colorcraft-dropdown", "colorcraft-mask-dropdown"])                               
                        modifier_active_widgets.append(w["active"])  
                        mod_mask_widgets.append(w["mask"])                      
                        gr_modifier_targeted = gr.Checkbox(value=False, visible=False, elem_classes=["colorcraft-targeted"], min_width=60)
                        modifier_targeted_checkboxes.append(gr_modifier_targeted)
                        all_components.append(gr_modifier_targeted)
                        gr_reset = gr.Button("\u2716", elem_classes=["colorcraft-reset"], min_width=10, size="sm")
                        with gr.Row(elem_classes=["colorcraft-schedule-row"]):
                            w["strength"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=1.0, label="Strength", min_width=60,
                                                       elem_classes=["colorcraft-strength"])
                            w["start"] = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Start", min_width=60,
                                                    elem_classes=["colorcraft-start"])
                            w["end"] = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="End", min_width=60,
                                                  elem_classes=["colorcraft-end"])
                        with InputAccordion(False, label="Advanced Schedule",
                                             elem_classes=["colorcraft-advanced-toggle"]) as w_advanced:
                            w["advanced"] = w_advanced
                            with gr.Row(elem_classes=["colorcraft-adv-row"]):
                                w["exponent"] = gr.Slider(minimum=0.0, maximum=3.0, step=0.01, value=0.0, label="Exponent", min_width=60,
                                                           elem_classes=["colorcraft-exponent"])
                                w["bias"] = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Bias", min_width=60,
                                                       elem_classes=["colorcraft-bias"])
                                w["start_off"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Start Offset", min_width=60,
                                                            elem_classes=["colorcraft-start-off"])
                                w["end_off"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="End Offset", min_width=60,
                                                          elem_classes=["colorcraft-end-off"])
                                w["smooth"] = gr.Checkbox(value=True, label="Smooth", min_width=60,
                                                           elem_classes=["colorcraft-smooth"])
                        with gr.Row(elem_classes=["colorcraft-column-row"]):
                            with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                with gr.Group(elem_classes=["colorcraft-group"]):
                                    w["exposure"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Exposure", min_width=60)
                                    w["tone_compression"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Tone Compression", min_width=60)                                    
                            with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                with gr.Group(elem_classes=["colorcraft-group"]):
                                    w["vibrance"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Vibrance", min_width=60)
                                    w["saturation"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Saturation", min_width=60)
                            with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                with gr.Group(elem_classes=["colorcraft-group"]):
                                    w["temperature"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Temperature", min_width=60, elem_classes=["colorcraft-temperature"])
                                    w["tint"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Tint", min_width=60, elem_classes=["colorcraft-tint"])
                        with gr.Row(elem_classes=["colorcraft-column-row"]): 
                            with gr.Column(min_width=90): 
                                with gr.Group(elem_classes=["colorcraft-group"]): 
                                    w["contrast"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Contrast", min_width=60)    
                            with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                with gr.Group(elem_classes=["colorcraft-group"]):
                                    w["chroma_contrast"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Chroma Contrast", min_width=60)
                                    w["chroma_center"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Chroma Center", min_width=60)
                            with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                with gr.Group(elem_classes=["colorcraft-group"]): 
                                    w["clarity"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Clarity", min_width=60)
                                    w["sharpness"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Sharpness", min_width=60)
                        with gr.Accordion("Chroma Plus", open=False):
                            with gr.Row(elem_classes=["colorcraft-column-row"]):
                                with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                    w["temp_plus_tint"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Temp+Tint", min_width=60, elem_classes=["colorcraft-temp-plus-tint"])
                                    w["temp_minus_tint"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Temp-Tint", min_width=60, elem_classes=["colorcraft-temp-minus-tint"])
                                with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                    w["lab_a"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Lab A", min_width=60, elem_classes=["colorcraft-lab-a"])
                                    w["lab_b"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Lab B", min_width=60, elem_classes=["colorcraft-lab-b"])
                                with gr.Column(elem_classes=["colorcraft-column"], min_width=90):
                                    w["lab_a_plus_b"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Lab A+B", min_width=60, elem_classes=["colorcraft-lab-a-plus-b"])
                                    w["lab_a_minus_b"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Lab A-B", min_width=60, elem_classes=["colorcraft-lab-a-minus-b"])
                        with gr.Accordion("Color Shift", open=False):
                            with gr.Column(elem_classes=["colorcraft-row-column"]):
                                with gr.Row(elem_classes=["colorcraft-row"]):
                                    w["color_shift_mode"] = gr.Dropdown(choices=COLOR_SHIFT_MODE_OPTIONS, value="default", label="Mode", min_width=60, elem_classes=["colorcraft-dropdown"]) 
                                    w["color_shift_amount"] = gr.Slider(minimum=-4.0, maximum=4.0, step=0.01, value=0.0, label="Amount", min_width=60)
                                    w["color_shift_brightness"] = gr.Slider(minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Brightness", min_width=60)
                            with gr.Column(elem_classes=["colorcraft-row-column"]):
                                with gr.Row(elem_classes=["colorcraft-row"]):
                                    w["color_shift_red"] = gr.Slider(minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Red", min_width=60)
                                    w["color_shift_green"] = gr.Slider(minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Green", min_width=60)
                                    w["color_shift_blue"] = gr.Slider(minimum=-2.0, maximum=2.0, step=0.01, value=0.0, label="Blue", min_width=60)

                        reset_fields = [f for f in MODIFIER_FIELDS if f != "active"]  # resetting values shouldn't also toggle the tab off, but mask selection DOES reset
                        reset_defaults = tuple(w[f].value for f in reset_fields)
                        gr_reset.click(fn=lambda vals=reset_defaults: vals, inputs=[], outputs=[w[f] for f in reset_fields], show_progress=False)

                        tab_widgets = [w[f] for f in MODIFIER_FIELDS]
                        modifier_components += tab_widgets
                        all_components += tab_widgets

                        self.infotext_fields.append((w["active"], (lambda d, tag=tag: _extract(d, "modifiers", tag, "active_flag", False))))
                        for f in MODIFIER_VALUE_FIELDS:
                            self.infotext_fields.append((w[f], (lambda d, tag=tag, f=f: _extract(d, "modifiers", tag, f))))

            with InputAccordion(False, label="Masking", elem_classes=["colorcraft-mask-accordion"]) as gr_masking_enabled:
                all_components.append(gr_masking_enabled)
                # Not written to infotext directly, but restores to ON
                # whenever real masks/combos exist in the parsed infotext.
                self.infotext_fields.append((gr_masking_enabled,
                    lambda d: bool(d.get("Colorcraft", {}).get("masks")) or bool(d.get("Colorcraft", {}).get("combos"))))

                with gr.Group(elem_classes=["colorcraft-tab-group", "colorcraft-mask-tab-group"]):
                    for i in range(MASK_COUNT):
                        tag = f"M{i+1}"
                        with gr.Tab(tag, elem_classes=["colorcraft-tab"]):
                            gr_reset = gr.Button("\u2716", elem_classes=["colorcraft-reset"], min_width=10, size="sm")
                            gr_leaf_active = gr.Checkbox(value=False, visible=False, elem_classes=["colorcraft-active"], min_width=60)
                            w = {}
                            with gr.Row(elem_classes=["colorcraft-column-row"]):
                                with gr.Column(elem_classes=["colorcraft-column"], min_width=60):
                                    w["mask_axis"] = gr.Dropdown(choices=MASK_AXIS_OPTIONS, value=MASK_AXIS_OPTIONS[0], label="Mask Axis", min_width=60,
                                                                  elem_classes=["colorcraft-dropdown", "colorcraft-mask-axis"])
                                    w["mask_strength"] = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Strength", min_width=60,
                                                                    elem_classes=["colorcraft-mask-strength"])
                                    w["mask_center"] = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.0, label="Center", min_width=60,
                                                                  elem_classes=["colorcraft-mask-center"])
                                    w["blur"] = gr.Slider(minimum=0.0, maximum=64.0, step=0.1, value=0.0, label="Blur", min_width=60)
                                with gr.Column(elem_classes=["colorcraft-column"], min_width=60):
                                    w["mask_mode"] = gr.Dropdown(choices=MASK_MODE_OPTIONS, value="highs", label="Mask Mode", min_width=60,
                                                                  elem_classes=["colorcraft-dropdown", "colorcraft-mask-mode"])
                                    w["mask_hardness"] = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=1.0, label="Hardness", min_width=60,
                                                                    elem_classes=["colorcraft-mask-hardness"])
                                    w["mask_width"] = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.0, label="Width", min_width=60,
                                                                 elem_classes=["colorcraft-mask-width"])
                                    w["spread"] = gr.Slider(minimum=-3.0, maximum=3.0, step=0.01, value=0.0, label="Spread", min_width=60)                                     
                                with gr.Column(elem_classes=["colorcraft-mask-plot-column"], min_width=60):                                    
                                    gr.HTML('<canvas class="colorcraft-mask-plot"></canvas>')
                                    w["contrast"] = gr.Slider(minimum=-10.0, maximum=10.0, step=0.01, value=0.0, label="Contrast", min_width=60)

                            reset_defaults = tuple(w[f].value for f in LEAF_FIELDS)
                            gr_reset.click(fn=lambda vals=reset_defaults: vals, inputs=[], outputs=[w[f] for f in LEAF_FIELDS], show_progress=False)

                            tab_widgets = [w[f] for f in LEAF_FIELDS]
                            leaf_components += tab_widgets
                            all_components += tab_widgets + [gr_leaf_active]
                            leaf_active_checkboxes.append(gr_leaf_active)

                            for f in LEAF_FIELDS:
                                self.infotext_fields.append((w[f], (lambda d, tag=tag, f=f: _extract(d, "masks", tag, f))))

                with gr.Group(elem_classes=["colorcraft-tab-group", "colorcraft-combo-tab-group"]):
                    for i in range(COMBO_COUNT):
                        tag = f"C{i+1}"
                        with gr.Tab(tag, elem_classes=["colorcraft-tab"]):
                            gr_reset = gr.Button("\u2716", elem_classes=["colorcraft-reset"], min_width=10, size="sm")
                            gr_combo_active = gr.Checkbox(value=False, visible=False, elem_classes=["colorcraft-active"], min_width=60)
                            gr_combo_label = gr.Textbox(value=tag, visible=False, elem_classes=["colorcraft-combo-label"])
                            w = {}
                            with gr.Column(elem_classes=["colorcraft-row-column"]):
                                with gr.Row(elem_classes=["colorcraft-row"]):
                                    ab_choices = ["none"] + [f"M{j+1}" for j in range(MASK_COUNT)] + [f"C{j+1}" for j in range(i)]
                                    w["mask_a"] = gr.Dropdown(choices=ab_choices, value="none", label="Mask A", min_width=60, elem_classes=["colorcraft-dropdown"])
                                    w["mask_b"] = gr.Dropdown(choices=ab_choices, value="none", label="Mask B", min_width=60, elem_classes=["colorcraft-dropdown"])
                                    w["operation"] = gr.Dropdown(choices=COMBO_OP_OPTIONS, value="and", label="Operation", min_width=60, elem_classes=["colorcraft-dropdown"])
                                with gr.Row(elem_classes=["colorcraft-row"]):
                                    w["blur"] = gr.Slider(minimum=0.0, maximum=64.0, step=0.1, value=0.0, label="Blur", min_width=60)
                                    w["spread"] = gr.Slider(minimum=-3.0, maximum=3.0, step=0.01, value=0.0, label="Spread", min_width=60)
                                    w["contrast"] = gr.Slider(minimum=-10.0, maximum=10.0, step=0.01, value=0.0, label="Contrast", min_width=60)

                            reset_defaults = tuple(w[f].value for f in COMBO_FIELDS)
                            gr_reset.click(fn=lambda vals=reset_defaults: vals, inputs=[], outputs=[w[f] for f in COMBO_FIELDS], show_progress=False)

                            tab_widgets = [w[f] for f in COMBO_FIELDS]
                            combo_components += tab_widgets
                            all_components += tab_widgets + [gr_combo_active]
                            combo_a_widgets.append(w["mask_a"])
                            combo_b_widgets.append(w["mask_b"])
                            combo_active_checkboxes.append(gr_combo_active)
                            combo_op_widgets.append(w["operation"])
                            combo_label_widgets.append(gr_combo_label)

                            for f in COMBO_FIELDS:
                                self.infotext_fields.append((w[f], (lambda d, tag=tag, f=f: _extract(d, "combos", tag, f))))

            with InputAccordion(False, label="Debug", elem_classes=["colorcraft-debug-accordion"]) as gr_debug_enabled:
                all_components.append(gr_debug_enabled)
                with gr.Row():
                    gr_debug_axis_style = gr.Dropdown(choices=DEBUG_AXIS_STYLES, value="colormap",
                                                       label="Axis Projection Style", elem_classes=["colorcraft-dropdown"])
                    gr_debug_composite_color = gr.Dropdown(choices=["none"] + list(DEBUG_COMPOSITE_COLORS.keys()), value="none",
                                                            label="Composite Mask Color", elem_classes=["colorcraft-dropdown"])
                    gr_debug_overlay_color = gr.Dropdown(choices=DEBUG_OVERLAY_COLORS, value="white", label="Overlay Color", elem_classes=["colorcraft-dropdown"])
                    gr_debug_step = gr.Slider(minimum=0, maximum=50, step=1, value=5, label="Debug Step")
                gr_debug_axes = gr.CheckboxGroup(choices=DEBUG_AXIS_OPTIONS, value=[], label="Axis Projections", elem_classes=["colorcraft-debug-group"], show_label=False)
                gr_debug_masks = gr.CheckboxGroup(choices=[f"M{i+1}" for i in range(MASK_COUNT)], value=[], label="Masks", elem_classes=["colorcraft-debug-group"], show_label=False)
                gr_debug_combos = gr.CheckboxGroup(choices=[f"C{i+1}" for i in range(COMBO_COUNT)], value=[], label="Combos", elem_classes=["colorcraft-debug-group"], show_label=False)                
                    
                all_components += [gr_debug_axes, gr_debug_masks, gr_debug_combos, gr_debug_composite_color,
                                    gr_debug_overlay_color, gr_debug_axis_style, gr_debug_step]

        topology_inputs = mod_mask_widgets + combo_a_widgets + combo_b_widgets + modifier_active_widgets + [gr_masking_enabled]
        topology_outputs = leaf_active_checkboxes + combo_active_checkboxes + mod_mask_widgets + modifier_targeted_checkboxes
        for wgt in topology_inputs:
            wgt.change(fn=on_topology_change, inputs=topology_inputs, outputs=topology_outputs, show_progress=False)

        label_inputs = combo_a_widgets + combo_b_widgets + combo_op_widgets
        for wgt in label_inputs:
            wgt.change(fn=on_combo_label_change, inputs=label_inputs, outputs=combo_label_widgets, show_progress=False)

        all_components += combo_label_widgets
        for c in all_components:
            c.do_not_save_to_config = True

        self.n_modifier_fields = len(MODIFIER_FIELDS)
        self.n_leaf_fields = len(LEAF_FIELDS)
        self.n_combo_fields = len(COMBO_FIELDS)
        return [gr_enabled] + modifier_components + [gr_masking_enabled] + leaf_components + combo_components + [
            gr_debug_enabled, gr_debug_axes, gr_debug_masks, gr_debug_combos, gr_debug_composite_color, gr_debug_overlay_color, gr_debug_axis_style, gr_debug_step]

    # -----------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------

    def process(self, p, enabled, *args):
        if not enabled:
            if hasattr(self, "callbacks_added"):
                self.remove_callbacks()
                delattr(self, "callbacks_added")
            return

        nm, nl, nc = self.n_modifier_fields, self.n_leaf_fields, self.n_combo_fields
        mod_args = args[0:nm * MODIFIER_COUNT]
        masking_enabled = args[nm * MODIFIER_COUNT]
        rest_start = nm * MODIFIER_COUNT + 1
        leaf_args_flat = args[rest_start:rest_start + nl * MASK_COUNT]
        combo_start = rest_start + nl * MASK_COUNT
        combo_args_flat = args[combo_start:combo_start + nc * COMBO_COUNT]
        debug_start = combo_start + nc * COMBO_COUNT
        debug_enabled = args[debug_start]
        debug_axes, debug_masks, debug_combos, debug_composite_color, debug_overlay_color, debug_axis_style, debug_step = \
            args[debug_start + 1:debug_start + 8]

        leaf_args = [tuple(leaf_args_flat[i * nl:(i + 1) * nl]) for i in range(MASK_COUNT)]
        combo_args = [tuple(combo_args_flat[i * nc:(i + 1) * nc]) for i in range(COMBO_COUNT)]
        mod_mask_selections = [mod_args[i * nm + 1] for i in range(MODIFIER_COUNT)]  # MODIFIER_FIELDS[1] == "mask"
        modifier_active_list = [mod_args[i * nm] for i in range(MODIFIER_COUNT)]  # MODIFIER_FIELDS[0] == "active"

        modifier_mask_map, leaf_specs, combo_specs = build_all_mask_specs(leaf_args, combo_args, mod_mask_selections)
        if not masking_enabled:
            # Masking accordion off -- functionally the same as if every
            # modifier's mask selection were "none": nothing actually gates
            # any modifier's edit. leaf_specs/combo_specs are left untouched
            # so Debug can still preview a mask's shape even while masking
            # itself is disabled for the actual generation.
            modifier_mask_map = {}

        # Infotext write -- tag-indexed, only entries actually in use
        # (active modifiers whose mask selection is also active; masks/
        # combos the topology says are active).
        combo_a_list = [dict(zip(COMBO_FIELDS, t))["mask_a"] for t in combo_args]
        combo_b_list = [dict(zip(COMBO_FIELDS, t))["mask_b"] for t in combo_args]
        leaf_active, combo_active = compute_mask_activeness(mod_mask_selections, combo_a_list, combo_b_list, modifier_active_list)

        mod_entries = []
        self.current_step = 0
        self.actual_steps = 0
        self.modifier_data = []
        needs_neutral_anchor = False

        for i in range(MODIFIER_COUNT):
            chunk = mod_args[i * nm:(i + 1) * nm]
            d = dict(zip(MODIFIER_FIELDS, chunk))
            if not d["active"]:
                continue
            if not d["advanced"]:
                d["bias"], d["exponent"], d["start_off"], d["end_off"] = 0.5, 0.0, 0.0, 0.0
            if d["contrast"] != 0:
                needs_neutral_anchor = True

            tag = ROMAN[i]
            mod_entries.append((tag, {**d, "active_flag": True}))

            self.modifier_data.append({
                "name": tag,
                "schedule": None,
                "schedule_params": dict(
                    start=d["start"], end=d["end"], bias=d["bias"], amount=d["strength"],
                    exponent=d["exponent"], start_off=d["start_off"], end_off=d["end_off"], smooth=d["smooth"],
                ),
                "params": d,
                "color_anchor": None,  # built below, only if color_shift_amount != 0
                "mask_spec": modifier_mask_map.get(tag),
            })

        leaf_entries = [(f"M{i+1}", dict(zip(LEAF_FIELDS, leaf_args[i]))) for i in range(MASK_COUNT) if leaf_active[i]]
        combo_entries = [(f"C{i+1}", dict(zip(COMBO_FIELDS, combo_args[i]))) for i in range(COMBO_COUNT) if combo_active[i]]
        if mod_entries:
            mod_str = write_entries(mod_entries, MODIFIER_VALUE_FIELDS)
            if masking_enabled:
                mask_str = write_entries(leaf_entries, LEAF_FIELDS) if leaf_entries else ""
                combo_str = write_entries(combo_entries, COMBO_FIELDS) if combo_entries else ""
            else:
                mask_str, combo_str = "", ""
            p.extra_generation_params["Colorcraft"] = combine_infotext_sections(mod_str, mask_str, combo_str)

        self.family = resolve_family(p)
        self.cur_basis = None
        if self.family:
            basis = load_basis(self.family, VECTORS_DIR)
            if basis is not None:
                self.cur_basis = {k: v.to(device=device) for k, v in basis.items()}
            else:
                print(f"[Colorcraft] WARNING: no colorcraft-{self.family}.safetensors found in "
                      f"{VECTORS_DIR}; vector-based controls disabled this run -- only contrast/color_shift will work.")

        self.dev = resolve_dev({}, self.family) if self.cur_basis is not None else None

        # Debug panel state -- leaf_specs/combo_specs are already built
        # for every tab regardless of activeness, so this just stores
        # them for the callback to look up a selected tag by name.
        self.debug_axes = list(debug_axes) if debug_enabled else []
        self.debug_masks = list(debug_masks) if debug_enabled else []
        self.debug_combos = list(debug_combos) if debug_enabled else []
        self.debug_composite_color = debug_composite_color
        self.debug_overlay_color = debug_overlay_color
        self.debug_axis_style = debug_axis_style
        self.debug_step = debug_step
        self.leaf_specs = leaf_specs
        self.combo_specs = combo_specs
        self.debug_images = {}
        self.debug_curve_info = {}
        self.debug_hue_chroma_frac = None
        self.debug_composite_base_latent = None
        self.debug_composite_base_latent_is_5d = False
        if (self.debug_axes or self.debug_masks or self.debug_combos) and self.cur_basis is None:
            print("[Colorcraft] WARNING: Debug panel selections present but no basis matched this "
                  "model -- nothing will be captured this run.")

        # Anchors built here, not lazily inside denoised_callback -- `p`
        # (needed for p.sd_model) is a real argument here but NOT reliably
        # available inside the callback, only `params` is.
        self.neutral_anchor = build_color_latent_forge(p, 0.0, 0.0, 0.0, 0.0) if needs_neutral_anchor else None
        for layer in self.modifier_data:
            pr = layer["params"]
            if pr["color_shift_amount"] != 0:
                layer["color_anchor"] = build_color_latent_forge(
                    p, pr["color_shift_red"], pr["color_shift_green"], pr["color_shift_blue"], pr["color_shift_brightness"])

        if not hasattr(self, "callbacks_added"):
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_after_cfg(self.denoised_callback)
            self.callbacks_added = True

    def before_process_batch(self, p, *args, **kwargs):
        self.current_step = 0
        self.actual_steps = 0

    def postprocess(self, p, processed, *args):
        if hasattr(self, "callbacks_added"):
            self.remove_callbacks()
            delattr(self, "callbacks_added")

        # Known v1 limitation: debug_images is reset at the start of every
        # process() call, but postprocess() only runs once for the whole
        # job -- with n_iter > 1, only the LAST iteration's captures
        # survive to be appended here, not every iteration's.
        debug_images = getattr(self, "debug_images", None)
        if debug_images:
            base_images = list(processed.images)  # snapshot BEFORE appending anything below
            snapshot_latent = getattr(self, "debug_composite_base_latent", None)
            if snapshot_latent is not None:
                # Debug captures the pre-edit state at the selected step
                # (see denoised_callback) -- processed.images only ever
                # reflects the true post-edit final output, which is never
                # what compositing should show, even at the final step.
                base_images = decode_debug_latent(p, snapshot_latent, p.width, p.height,
                                                   is_5d=getattr(self, "debug_composite_base_latent_is_5d", False))
            total = sum(t.shape[0] for t in debug_images.values())
            if total > 20:
                print(f"[Colorcraft] WARNING: Debug panel producing {total} extra images this run "
                      f"(batch size x selected axes/masks/combos) -- consider narrowing the selection.")
            composite_color = DEBUG_COMPOSITE_COLORS.get(getattr(self, "debug_composite_color", "none"))
            overlay_color = getattr(self, "debug_overlay_color", "red")
            axis_style = getattr(self, "debug_axis_style", "greyscale")
            for label, tensor in debug_images.items():
                is_projection = label.startswith("axis:")
                curve_infos = self.debug_curve_info.get(label)
                tag = label.split(":", 1)[1] if ":" in label else None
                if tag in ("hue", "hue (weighted)"):
                    chroma_frac = self.debug_hue_chroma_frac if tag == "hue (weighted)" else None
                    imgs = render_hue_images(tensor, p.width, p.height, label, overlay_color, curve_infos, chroma_frac)
                    processed.images.extend(imgs)
                    continue
                if label.startswith("mask:"):
                    spec = self.leaf_specs.get(tag)
                elif label.startswith("combo:"):
                    spec = self.combo_specs.get(tag)
                else:
                    spec = None
                # Checked directly against every leaf in the spec (not just
                # curve_infos, which excludes hue-axis leaves) -- a hue-axis
                # split-mode leaf can still legitimately go negative even
                # though it doesn't get a curve drawn.
                is_signed = spec is not None and any(leaf.get("mask_mode") == "split" for leaf in collect_leaves(spec))
                if not is_projection and composite_color is not None:
                    imgs = composite_mask_images(tensor, composite_color, base_images, label, overlay_color, curve_infos)
                else:
                    imgs = debug_tensor_to_images(tensor, p.width, p.height, is_projection, is_signed, label,
                                                   overlay_color, curve_infos, axis_style)
                processed.images.extend(imgs)
            self.debug_images = {}

    def remove_callbacks(self):
        remove_callbacks_for_function(self.denoiser_callback)
        remove_callbacks_for_function(self.denoised_callback)

    def denoiser_callback(self, params):
        step = max(params.sampling_step, params.denoiser.step)
        steps = max(params.total_sampling_steps, params.denoiser.total_steps)
        actual_steps = steps - max(steps // params.denoiser.steps - 1, 0)
        self.current_step = min(step, actual_steps - 1)
        self.actual_steps = actual_steps
        if self.current_step == 0:
            self.debug_step = min(self.debug_step, actual_steps - 1)

    def denoised_callback(self, params):
        x = params.x
        is_5d = x.dim() == 5
        if is_5d:
            x = x.squeeze(2)

        # Debug capture happens BEFORE the modifier loop, using x exactly
        # as this step received it, before any modifier's edit -- matches
        # what apply_mask_gate itself evaluates against (each layer's own
        # pre-edit state). Always captures a fresh snapshot, even at the
        # final step, since post-edit processed.images doesn't match the
        # pre-edit state debug shows.
        if self.current_step == self.debug_step and self.cur_basis is not None:
            if self.debug_composite_color != "none":
                self.debug_composite_base_latent = downscale_latent_for_storage(x, max_dim=64)
                self.debug_composite_base_latent_is_5d = is_5d
            for axis in self.debug_axes:
                if axis in ("hue", "hue (weighted)"):
                    axis1, axis2 = chroma_axes(self.cur_basis, self.dev["chroma_plane"])
                    proj = compute_hue_projection(x, axis1, axis2, self.dev.get("hue_bias", 0.0))
                    self.debug_images[f"axis:{axis}"] = proj.detach().cpu()
                    if axis == "hue (weighted)":
                        sat_proj = compute_saturation_projection(x, axis1, axis2, self.dev["max_chroma"])
                        self.debug_hue_chroma_frac = ((sat_proj + 1.0) / 2.0).clamp(0.0, 1.0).detach().cpu()
                    continue
                proj = compute_axis_projection(axis, x, self.cur_basis, self.dev)
                if proj is not None:
                    self.debug_images[f"axis:{axis}"] = proj.detach().cpu()
            for tag in self.debug_masks:
                spec = self.leaf_specs.get(tag)
                if spec is not None:
                    self.debug_images[f"mask:{tag}"] = resolve_mask_tensor(spec, x, self.cur_basis, self.dev, VAE_DOWNSCALE_FACTOR).detach().cpu()
                    curve_infos = build_curve_infos(spec, x, self.cur_basis, self.dev)
                    if curve_infos:
                        self.debug_curve_info[f"mask:{tag}"] = [dict(ci, projection=ci["projection"].detach().cpu()) for ci in curve_infos]
            for tag in self.debug_combos:
                spec = self.combo_specs.get(tag)
                if spec is None:
                    print(f"[Colorcraft] WARNING: Debug combo {tag} has an unresolved Mask A/B reference -- skipping capture.")
                    continue
                self.debug_images[f"combo:{tag}"] = resolve_mask_tensor(spec, x, self.cur_basis, self.dev, VAE_DOWNSCALE_FACTOR).detach().cpu()
                curve_infos = build_curve_infos(spec, x, self.cur_basis, self.dev)
                if curve_infos:
                    self.debug_curve_info[f"combo:{tag}"] = [dict(ci, projection=ci["projection"].detach().cpu()) for ci in curve_infos]

        for layer in self.modifier_data:
            if layer["schedule"] is None:
                layer["schedule"] = make_schedule(self.actual_steps, **layer["schedule_params"])
            s = layer["schedule"][self.current_step]
            # print(f"[Colorcraft] {layer['name']} -- step {self.current_step}/{self.actual_steps - 1} -- multiplier: {s:.4f}")
            if s == 0:
                continue

            pr = layer["params"]
            pre = x
            out = pre

            if self.cur_basis is not None:
                axis1, axis2 = self.cur_basis["temperature"], self.cur_basis["tint"]  # chroma_plane fixed to temp/tint (dev override dropped)
                out = apply_vector_offset(out, self.cur_basis["exposure"], s * pr["exposure"])
                out = apply_tone_compression(out, self.cur_basis["exposure"], s * pr["tone_compression"])

            if pr["contrast"] != 0:
                out = apply_contrast(out, s * pr["contrast"], self.neutral_anchor)

            if self.cur_basis is not None:
                out = apply_vibrance(out, axis1, axis2, s * pr["vibrance"] * 2.0, k=self.dev["vibrance_k"], recenter=self.dev["recenter"], r_max=self.dev["max_chroma"])
                out = apply_vibrance(out, axis1, axis2, s * pr["saturation"], k=0.0, r_max=0.0, recenter=self.dev["recenter"])
                out = apply_vector_offset(out, self.cur_basis["temperature"], s * pr["temperature"])
                out = apply_vector_offset(out, self.cur_basis["tint"], s * pr["tint"])
                out = apply_vector_offset(out, self.cur_basis["temp+tint"], s * pr["temp_plus_tint"])
                out = apply_vector_offset(out, self.cur_basis["temp-tint"], s * pr["temp_minus_tint"])
                out = apply_vector_offset(out, self.cur_basis["lab-a"], s * pr["lab_a"])
                out = apply_vector_offset(out, self.cur_basis["lab-b"], s * pr["lab_b"])
                out = apply_vector_offset(out, self.cur_basis["lab-a+b"], s * pr["lab_a_plus_b"])
                out = apply_vector_offset(out, self.cur_basis["lab-a-b"], s * pr["lab_a_minus_b"])
                out = apply_chroma_contrast(out, axis1, axis2, s * pr["chroma_contrast"], r_max=self.dev["max_chroma"], chroma_center=pr["chroma_center"], recenter=self.dev["recenter"])

            if pr["color_shift_amount"] != 0 and layer["color_anchor"] is not None:
                out = apply_color_shift(out, s * pr["color_shift_amount"], pr["color_shift_mode"], layer["color_anchor"])

            if self.cur_basis is not None:
                out = apply_vector_offset(out, self.cur_basis["clarity"], s * pr["clarity"])
                out = apply_vector_offset(out, self.cur_basis["sharpness"], s * pr["sharpness"])

            if layer["mask_spec"] is not None and self.cur_basis is not None:
                out = apply_mask_gate(pre, out, layer["mask_spec"], self.dev, self.cur_basis, VAE_DOWNSCALE_FACTOR)

            x = out

        if is_5d:
            x = x.unsqueeze(2)
        params.x = x
