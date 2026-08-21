import os
import numpy as np
import torch
from PIL import Image
import comfy.samplers
import comfy.model_patcher
import comfy.model_management

from .lib_colorcraft.schedule import make_schedule
from .lib_colorcraft.vectors import (
    apply_vector_offset, apply_vector_scale, apply_vibrance,
    apply_chroma_contrast, apply_tone_compression, chroma_axes,
)
from .lib_colorcraft.masking import (
    MASK_AXIS_OPTIONS, compute_mask, compute_hue_mask, compute_saturation_mask,
    gaussian_blur_mask, apply_mask_spread, resolve_mask_tensor, apply_mask_gate, compute_hue_projection,
)
from .lib_colorcraft.color import (
    apply_contrast, apply_color_shift, build_color_latent, to_model_space,
)
from .lib_colorcraft.basis import (
    BASIS_FAMILIES, VAE_DOWNSCALE_FACTOR, MODEL_DEV_DEFAULTS, LATENT_FORMAT_TO_FAMILY,
    load_basis, resolve_dev,
)
from .lib_colorcraft.debug import (
    DEBUG_COMPOSITE_COLORS, DEBUG_OVERLAY_COLORS, DEBUG_AXIS_STYLES,
    compute_axis_projection, debug_tensor_to_images, composite_mask_images, render_hue_images,
    collect_leaves, build_curve_infos,
)


# Step-position lookup, Comfy-specific (sigma-based) -- Forge just reads
# a real step index off its callback params instead.

def sigma_to_value(sigma, sigmas, schedule):
    """Maps the sigma a model eval actually ran at to a value from the
    discrete per-step schedule array. Adapted from Jonseed's ComfyUI
    port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon"""
    real_sigmas = sigmas[:-1]
    n = len(schedule)
    if n < 2 or len(real_sigmas) < 2 or sigma <= 0:
        return float(schedule[0]) if n else 0.0

    deltas = (real_sigmas - sigma).abs()
    idx = int(deltas.argmin())

    if (
        (idx == 0 and sigma >= real_sigmas[0])
        or (idx == n - 1 and sigma <= real_sigmas[-1])
        or deltas[idx] == 0
    ):
        return float(schedule[idx])

    idx_lo, idx_hi = (idx, idx - 1) if sigma > real_sigmas[idx] else (idx + 1, idx)
    sig_lo, sig_hi = real_sigmas[idx_lo], real_sigmas[idx_hi]
    if sig_hi == sig_lo:
        return float(schedule[idx_lo])
    ratio = float(max(0.0, min(1.0, (sigma - sig_lo) / (sig_hi - sig_lo))))
    return float(schedule[idx_lo] + ratio * (schedule[idx_hi] - schedule[idx_lo]))

# Basis vectors keyed by VAE family, not by any specific diffusion
# model -- Krea2/QwenImage share one bundle, Flux/Z-Image share another.
# All families load once per sampler run regardless of what's connected.

VECTORS_DIR = os.path.join(os.path.dirname(__file__), "vectors")


# ---------------------------------------------------------------------------
# ColorcraftBasic -- the basic node. No vectors, no masking; a wildcard that
# runs and should work regardless of which real model the sampler resolves to.
# ---------------------------------------------------------------------------

class ColorcraftBasic:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # -- schedule ------------------------------------------------------
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks; never read
                # server-side, since the sampler already knows the real step count.
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),

                # -- contrast --------------------------------------------------------
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- color_shift -------------------------------------------------------
                # No accordion here (node's small enough not to need one) and no
                # separate on/off widget -- gated on color_shift_amount != 0, the
                # same "0 = no-op" convention as everything else on this node.
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
            },
        }

    def make(self, modifiers=None, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "basic", "params": params})
        return (chain,)


# ---------------------------------------------------------------------------
# ColorcraftAdvanced — the mega-node. Non-basic sliders here depend
# on whatever VAE-family basis the sampler resolves against the actual
# connected VAE.
# ---------------------------------------------------------------------------

class ColorcraftAdvanced:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # -- schedule ------------------------------------------------------
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks;
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),

                # -- luma group ------------------------------------------------------
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tone_compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),

                # -- punch group -----------------------------------------------------
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clarity": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- chroma/color group -------------------------------------------------
                "temperature": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "chroma_contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "chroma_center": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                # -- chroma plus group -------------------------------------------------
                "more_colors": ("BOOLEAN", {"default": False}),
                "temp_plus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "temp_minus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_plus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_minus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # -- color_shift group (accordion) -------------------------------------
                "color_shift": ("BOOLEAN", {"default": False}),
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),

                # -- dev group (accordion) -----------------------------------------------
                # Per-model calibrated values live in MODEL_DEV_DEFAULTS; these three
                # are overrides for a power user who wants to deviate from the calibrated
                # values, each gated by its own *_override bool.
                "dev": ("BOOLEAN", {"default": False}),
                "recenter_override": ("BOOLEAN", {"default": False}),
                "recenter": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_chroma_override": ("BOOLEAN", {"default": False}),
                "max_chroma": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "chroma_plane_override": ("BOOLEAN", {"default": False}),
                "chroma_plane": (["temp_tint", "lab"],),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, modifiers=None, masking=None, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "advanced", "params": params, "mask": masking})
        return (chain,)


# ---------------------------------------------------------------------------
# Sub-module nodes -- the modular alternative to ColorcraftAdvanced. Any
# number chain together via `modifiers`. Each (except Masking, Schedule)
# optionally takes a `masking` input built by ColorcraftMasking (or a
# Combine/Blur tree on top of one). Luma/Chroma/Punch take a required
# `schedule` input built by ColorcraftSchedule instead of owning their own
# schedule widgets, so several can share one schedule.
# ---------------------------------------------------------------------------

class ColorcraftSchedule:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_SCHEDULE",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "advanced": ("BOOLEAN", {"default": False}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_off": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "smooth": ("BOOLEAN", {"default": True}),
                # UI-only -- purely for the JS schedule plot tick marks;
                "plot_steps": ("INT", {"default": 8, "min": 2, "max": 20, "step": 1}),
            },
        }

    def make(self, **params):
        if not params.pop("advanced", False):
            params["exponent"] = 0.0
            params["start_off"] = 0.0
            params["end_off"] = 0.0
        return (params,)


class ColorcraftLuma:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tone_compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "luma", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftChroma:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "temperature": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "chroma_contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "chroma_center": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "chroma", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftChromaPlus:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "temp_plus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "temp_minus_tint": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_plus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lab_a_minus_b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "chroma_plus", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftPunch:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clarity": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "punch", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftShift:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MODIFIER",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("COLORCRAFT_SCHEDULE",),
                "color_shift_amount": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["default", "legacy"],),
                "red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "masking": ("COLORCRAFT_MASK",),
            },
        }

    def make(self, schedule, modifiers=None, masking=None, **params):
        chain = list(modifiers) if modifiers else []
        chain.append({"kind": "shift", "params": params, "mask": masking, "schedule": schedule})
        return (chain,)


class ColorcraftMasking:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_mode": (["highs", "lows", "split", "range", "protect range"],),
                "mask_axis": (MASK_AXIS_OPTIONS,),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_width": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask_center": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "mask_hardness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    def make(self, **params):
        return (params,)


class ColorcraftMaskCombine:
    """Combines two COLORCRAFT_MASK specs via fuzzy set logic -- e.g. AND a
    temperature-highs mask with an exposure-highs mask to target only warm
    highlights. Chain multiple Combine nodes for 3+ masks. Just packages a
    spec; the actual math happens in resolve_mask_tensor at sampling time."""
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("COLORCRAFT_MASK",),
                "mask_b": ("COLORCRAFT_MASK",),
                "operation": (["and", "or", "subtract", "xor"],),
            },
        }

    def make(self, mask_a, mask_b, operation):
        return ({"operation": operation, "a": mask_a, "b": mask_b},)


class ColorcraftMaskBlur:
    """Spatially blurs a COLORCRAFT_MASK's eventual influence, so it spreads
    past the exact pixels that satisfied its gate condition. A separate node
    (not folded into ColorcraftMasking) so it can wrap any point in a
    mask chain, including a Combine result. Radius is in decoded-image
    pixels, converted internally by VAE_DOWNSCALE_FACTOR. `spread` — see
    apply_mask_spread. `contrast` — see apply_mask_contrast; meant to run
    AFTER spread (blur -> spread -> contrast), which resolve_mask_tensor's
    blur-node handling already does regardless of the order these widgets
    are declared in here."""
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("COLORCRAFT_MASK",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("COLORCRAFT_MASK",),
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 160.0, "step": 0.1}),
                "spread": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
            },
        }

    def make(self, mask, radius, spread, contrast):
        return ({"blur": radius, "spread": spread, "contrast": contrast, "a": mask},)


# ---------------------------------------------------------------------------
# ColorcraftSampler — wraps a SAMPLER. Loads every basis family (cheap, tiny
# files), then resolves once which family (if any) matches the connected
# model. Basic-node controls always work; vector-based controls get disabled
# with a console warning if nothing matches.
# ---------------------------------------------------------------------------

class ColorcraftSampler:
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("SAMPLER", "COLORCRAFT_DEBUG")
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "vae": ("VAE",),
            },
            "optional": {
                "modifiers": ("COLORCRAFT_MODIFIER",),
                "debug_step": ("INT", {"forceInput": True}),
            },
        }

    def wrap(self, sampler, vae, modifiers=None, debug_step=None):
        debug_container = {}  # mutated in place inside post_cfg_function during actual
        # sampling -- this node's outputs are fixed at graph-build time, before any
        # sampling runs, so this empty dict IS the output; a downstream debug node
        # reads it after the sampler that consumes this SAMPLER has executed.
        #
        # No early return when modifiers is empty -- debug capture only needs the
        # model's own resolved family/basis, not any modifier chain.
        modifiers = modifiers or []

        all_basis = {f: load_basis(f, VECTORS_DIR) for f in BASIS_FAMILIES}
        all_basis = {f: b for f, b in all_basis.items() if b is not None}

        color_cache = {}
        basis_cache = {}
        resolved = {"checked": False, "family": None}

        def get_color_latent(color, latent_format, device, dtype):
            key = tuple(round(c, 4) for c in color)
            if key not in color_cache:
                anchor = to_model_space(vae, latent_format, build_color_latent(vae, comfy.model_management.get_torch_device(), *color))
                color_cache[key] = anchor.to(device=device, dtype=dtype)
            return color_cache[key]

        def get_basis(family, device, dtype):
            key = (family, device, dtype)
            if key not in basis_cache:
                basis_cache[key] = {k: v.to(device=device, dtype=dtype) for k, v in all_basis[family].items()}
            return basis_cache[key]

        def resolve_family(latent_format, any_advanced):
            if resolved["checked"]:
                return resolved["family"]
            resolved["checked"] = True
            fmt_name = type(latent_format).__name__ if latent_format is not None else None
            family = LATENT_FORMAT_TO_FAMILY.get(fmt_name)
            if family is not None and family not in all_basis:
                print(f"[Colorcraft] WARNING: detected VAE family '{family}' (latent_format={fmt_name}) "
                      f"but no matching colorcraft-{family}.safetensors was found; vector-based controls "
                      f"disabled this run -- only Basic-node controls (contrast/color_shift) will work.")
                family = None
            elif family is None and any_advanced:
                print(f"[Colorcraft] WARNING: no basis matches the current model (latent_format={fmt_name}); "
                      f"vector-based controls (Advanced/Luma/Chroma/Chroma Plus/Punch/Masking) disabled this run "
                      f"-- Basic-node controls (contrast/color_shift) still work, and so does Shift's own "
                      f"color_shift effect, but any masking connected to Shift will silently have no effect.")
            resolved["family"] = family
            return family

        def wrapped_sampler_function(model, x, sigmas, *args, extra_args=None, **kwargs):
            extra_args = dict(extra_args or {})
            num_steps = len(sigmas) - 1

            # Kinds whose controls depend on the resolved basis vectors.
            ADVANCED_KINDS = {"advanced", "luma", "chroma", "chroma_plus", "punch"}

            built = []
            for entry in modifiers:
                p = entry["params"]
                # Basic/Advanced own their schedule widgets inline (in `p`);
                # Luma/Chroma/Punch get theirs from a required
                # COLORCRAFT_SCHEDULE input instead, stashed on the entry.
                sched = entry.get("schedule") or p
                schedule_kwargs = {k: sched[k] for k in
                                    ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth")}
                schedule = make_schedule(num_steps, amount=sched["strength"], **schedule_kwargs)
                built.append((schedule, entry["kind"], p, entry.get("mask")))

            # Shift works on any model and isn't in ADVANCED_KINDS, but its
            # optional masking input does need a basis -- warn only in that case.
            any_advanced = any(
                kind in ADVANCED_KINDS or (kind == "shift" and mask_params)
                for _, kind, _, mask_params in built
            )

            # post_cfg_function fires exactly once per sampling step, in
            # order -- a plain int can't be rebound from inside a nested
            # closure, so a mutable single-key dict stands in for one.
            debug_step_counter = {"i": 0}
            debug_step_target = min(debug_step, num_steps - 1) if debug_step is not None else None

            def post_cfg_function(pc_args):
                x0 = pc_args["denoised"]
                cur_sigma = pc_args["sigma"].max().item()
                latent_format = getattr(pc_args.get("model"), "latent_format", None)

                is_5d = x0.dim() == 5
                if is_5d:
                    x0 = x0.squeeze(2)

                family = resolve_family(latent_format, any_advanced)
                cur_basis = get_basis(family, x0.device, x0.dtype) if family else None

                # Debug capture happens before the modifier loop, using
                # x0 exactly as this callback received it -- matches what
                # apply_mask_gate itself evaluates against (each layer's
                # own pre-edit state).
                if debug_step_target is not None:
                    current_step = debug_step_counter["i"]
                    debug_step_counter["i"] += 1
                    if cur_basis is not None:
                        if current_step == debug_step_target:
                            debug_dev = resolve_dev({}, family)
                            target_w = x0.shape[-1] * VAE_DOWNSCALE_FACTOR
                            target_h = x0.shape[-2] * VAE_DOWNSCALE_FACTOR
                            debug_container["latent"] = x0.detach().cpu()
                            debug_container["target_width"] = target_w
                            debug_container["target_height"] = target_h
                            debug_container["cur_basis"] = {k: v.detach().cpu() for k, v in cur_basis.items()}
                            debug_container["dev"] = debug_dev

                out = x0
                for i, (schedule, kind, p, mask_params) in enumerate(built):
                    s = sigma_to_value(cur_sigma, sigmas, schedule)
                    if s == 0:
                        continue

                    pre = out
                    if kind != "advanced" and "contrast" in p and p["contrast"] != 0:
                        neutral_anchor = get_color_latent(
                            (0.0, 0.0, 0.0, 0.0), latent_format, out.device, out.dtype,
                        )
                        out = apply_contrast(out, s * p["contrast"], neutral_anchor)

                    if kind == "basic":
                        if p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

                    elif kind == "advanced":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                            out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                            out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])

                        if p["contrast"] != 0:
                            neutral_anchor = get_color_latent(
                                (0.0, 0.0, 0.0, 0.0), latent_format, out.device, out.dtype,
                            )
                            out = apply_contrast(out, s * p["contrast"], neutral_anchor)

                        if cur_basis is not None:
                            out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                            out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                            out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                            out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                            if p["more_colors"]:
                                out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                                out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                                out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                                out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                                out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                                out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                            out = apply_chroma_contrast(
                                out, axis1, axis2, s * p["chroma_contrast"],
                                r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                            )

                        if p["color_shift"] and p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                            out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])

                        # Reuses the same `dev` already resolved above for its own
                        # edits to gate an external mask -- one resolution, shared.
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                    elif kind == "luma":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                            out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                    elif kind == "chroma":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                            out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                            out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                            out = apply_chroma_contrast(
                                out, axis1, axis2, s * p["chroma_contrast"],
                                r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                            )
                            out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                            out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                    elif kind == "chroma_plus":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                            out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                            out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                            out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                            out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                            out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                    elif kind == "punch":
                        dev = resolve_dev(p, family) if cur_basis is not None else None
                        if cur_basis is not None:
                            out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                            out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])
                        if mask_params and cur_basis is not None:
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                    elif kind == "shift":
                        if p["color_shift_amount"] != 0:
                            color_anchor = get_color_latent(
                                (p["red"], p["green"], p["blue"], p["brightness"]),
                                latent_format, out.device, out.dtype,
                            )
                            out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)
                        if mask_params and cur_basis is not None:
                            dev = resolve_dev(p, family)
                            out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, VAE_DOWNSCALE_FACTOR)

                if is_5d:
                    out = out.unsqueeze(2)
                return out

            model_options = comfy.model_patcher.set_model_options_post_cfg_function(
                extra_args.get("model_options", {}), post_cfg_function,
            )
            extra_args["model_options"] = model_options

            return sampler.sampler_function(
                model, x, sigmas, *args, extra_args=extra_args, **kwargs
            )

        new_sampler = comfy.samplers.KSAMPLER(
            wrapped_sampler_function,
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (new_sampler, debug_container)



def comfy_image_batch_to_pil_list(batch):
    """[B,H,W,C] 0..1 float -> list of RGB PIL images, matching Comfy's
    own SaveImage/PreviewImage convention."""
    arr = (batch.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    return [Image.fromarray(arr[b], mode="RGB") for b in range(arr.shape[0])]


class ColorcraftDebug:
    """Takes a COLORCRAFT_DEBUG (from ColorcraftSampler) plus any number of
    COLORCRAFT_MASK specs -- including ones never wired into an actual
    modifier's masking input -- and bakes axis projections + mask
    previews into one IMAGE batch, ready for a PreviewImage node.

    No VAE decode happens here or in ColorcraftSampler: WanVAE-based
    models pay a real mid-generation cost swapping the live-preview
    TAEHV out for the real VAE. Compositing is optional instead -- an
    `image` input lets the user plug in an already-decoded image (their
    own SaveImage/PreviewImage source), gated purely on that input's
    presence, not a "none" toggle. Composite (multi-child) mask specs get
    the same histogram/overlay treatment as leaves, just without a
    curve."""
    CATEGORY = "Muerrilla/Colorcraft"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"

    DEBUG_AXES = [a for a in MASK_AXIS_OPTIONS if a != "hue"] + ["hue"]

    @classmethod
    def INPUT_TYPES(cls):
        axis_widgets = {f"axis_{a}": ("BOOLEAN", {"default": False}) for a in cls.DEBUG_AXES}
        return {
            "required": {
                "colorcraft_debug": ("COLORCRAFT_DEBUG",),
                # Purely an execution-order anchor -- COLORCRAFT_DEBUG alone
                # is satisfied the instant wrap() returns, before the real
                # sampling that actually populates it runs. Wire this from
                # anything downstream of the real sampling (the sampled
                # LATENT, the decoded IMAGE) to force this node to run
                # after it. The value itself is never read.
                "run_after": ("*", {"forceInput": True}),
                "composite_color": (list(DEBUG_COMPOSITE_COLORS.keys()),),
                "axis_style": (DEBUG_AXIS_STYLES,),
                "overlay_color": (DEBUG_OVERLAY_COLORS,),
                **axis_widgets,
            },
            "optional": {
                "image": ("IMAGE",),
                "mask_0": ("COLORCRAFT_MASK",),
            },
        }

    def render(self, colorcraft_debug, run_after, composite_color, axis_style, overlay_color, image=None, **kwargs):
        if not colorcraft_debug or colorcraft_debug.get("latent") is None:
            print("[Colorcraft] WARNING: Debug Preview has nothing to show -- either debug_step never "
                  "landed on a step that ran, or no basis matched the model this run.")
            return (torch.zeros(1, 64, 64, 3),)

        latent = colorcraft_debug["latent"]
        cur_basis = colorcraft_debug["cur_basis"]
        dev = colorcraft_debug["dev"]
        width, height = colorcraft_debug["target_width"], colorcraft_debug["target_height"]
        # Compositing is gated purely on the image input's presence --
        # composite_color only picks which color once that gate is open.
        base_images = comfy_image_batch_to_pil_list(image) if image is not None else None
        composite_rgb = DEBUG_COMPOSITE_COLORS[composite_color] if image is not None else None

        images = []
        for axis in self.DEBUG_AXES:
            if not kwargs.get(f"axis_{axis}"):
                continue
            if axis == "hue":
                axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                proj = compute_hue_projection(latent, axis1, axis2, dev.get("hue_bias", 0.0))
                images += render_hue_images(proj, width, height, "axis:hue", overlay_color)
                continue
            proj = compute_axis_projection(axis, latent, cur_basis, dev)
            if proj is None:
                continue
            images += debug_tensor_to_images(proj, width, height, True, False, f"axis:{axis}", overlay_color, axis_style=axis_style)

        mask_items = sorted(
            ((k, v) for k, v in kwargs.items() if k.startswith("mask_") and v is not None),
            key=lambda kv: int(kv[0].split("_")[1]),
        )
        for label, spec in mask_items:
            mask_tensor = resolve_mask_tensor(spec, latent, cur_basis, dev, VAE_DOWNSCALE_FACTOR)
            leaves = collect_leaves(spec)
            curve_infos = build_curve_infos(spec, latent, cur_basis, dev)
            is_signed = any(leaf.get("mask_mode") == "split" for leaf in leaves)
            if composite_rgb is not None:
                images += composite_mask_images(mask_tensor, composite_rgb, base_images, label, overlay_color, curve_infos or None)
            else:
                images += debug_tensor_to_images(mask_tensor, width, height, False, is_signed, label, overlay_color, curve_infos or None)

        if not images:
            return (torch.zeros(1, 64, 64, 3),)

        arrs = [np.array(im).astype(np.float32) / 255.0 for im in images]
        return (torch.from_numpy(np.stack(arrs, axis=0)),)


NODE_CLASS_MAPPINGS = {
    "ColorcraftBasic": ColorcraftBasic,
    "ColorcraftAdvanced": ColorcraftAdvanced,
    "ColorcraftLuma": ColorcraftLuma,
    "ColorcraftChroma": ColorcraftChroma,
    "ColorcraftChromaPlus": ColorcraftChromaPlus,
    "ColorcraftSchedule": ColorcraftSchedule,
    "ColorcraftPunch": ColorcraftPunch,
    "ColorcraftShift": ColorcraftShift,
    "ColorcraftMasking": ColorcraftMasking,
    "ColorcraftMaskCombine": ColorcraftMaskCombine,
    "ColorcraftMaskBlur": ColorcraftMaskBlur,
    "ColorcraftSampler": ColorcraftSampler,
    "ColorcraftDebug": ColorcraftDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorcraftBasic": "Colorcraft Basic",
    "ColorcraftAdvanced": "Colorcraft Advanced",
    "ColorcraftLuma": "Colorcraft Luma",
    "ColorcraftChroma": "Colorcraft Chroma",
    "ColorcraftChromaPlus": "Colorcraft Chroma Plus",
    "ColorcraftSchedule": "Colorcraft Schedule",
    "ColorcraftPunch": "Colorcraft Punch",
    "ColorcraftShift": "Colorcraft Shift",
    "ColorcraftMasking": "Colorcraft Masking",
    "ColorcraftMaskCombine": "Colorcraft Combine Masks",
    "ColorcraftMaskBlur": "Refine Mask",
    "ColorcraftSampler": "Colorcraft Sampler",
    "ColorcraftDebug": "Colorcraft Debug",
}