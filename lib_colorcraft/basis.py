import os

from safetensors.torch import load_file

BASIS_FAMILIES = ["krea2", "zimage"]

# Both supported VAE families downscale 8x -- used to convert
# ColorcraftMaskBlur's radius from decoded-image pixels (what the UI shows)
# to latent pixels (what gaussian_blur_mask operates on).
VAE_DOWNSCALE_FACTOR = 8

# latent_format class name -> basis family. Krea2/QwenImage report "Wan21";
# Flux/Z-Image report "Flux". Comfy reads the class name off latent_format
# directly; Forge reads it off p.sd_model.model_config.latent_format.
LATENT_FORMAT_TO_FAMILY = {
    "Wan21": "krea2",
    "Flux": "zimage",
}

# Per-model calibrated defaults for the chroma-plane math. vibrance_k/
# exposure_scale/color_scale/hue_bias are internal only, no UI override.
# recenter/max_chroma/chroma_plane are overridable on Advanced (see
# resolve_dev) as artistic controls. exposure_scale/color_scale normalize
# each axis's raw projection to roughly +-1 across models. hue_bias
# (radians) rotates zimage's hue angle to match krea2's calibration.
MODEL_DEV_DEFAULTS = {
    "krea2":  {"vibrance_k": 1.0, "max_chroma": 2.5, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 3.5, "color_scale": 3.0, "detail_scale": 4.0, "hue_bias": -0.1},
    "zimage": {"vibrance_k": 1.5, "max_chroma": 5.0, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 7.5, "color_scale": 6.0, "detail_scale": 4.0, "hue_bias": -0.4},
}


def load_basis(family, vectors_dir):
    """Loads colorcraft-<family>.safetensors from vectors_dir. Returns
    dict[name -> 1D tensor] or None if missing. vectors_dir must be
    supplied by the caller, computed relative to its own entry-point
    file -- resolving it from this module's own __file__ would point at
    lib_colorcraft/ instead of the repo root."""
    path = os.path.join(vectors_dir, f"colorcraft-{family}.safetensors")
    if not os.path.isfile(path):
        return None
    return load_file(path)


def resolve_dev(params, family):
    """Resolves recenter/max_chroma/chroma_plane against
    MODEL_DEV_DEFAULTS, honoring per-field *_override flags in params
    when present (only Advanced exposes these)."""
    dev = MODEL_DEV_DEFAULTS[family]
    return {
        "vibrance_k": dev["vibrance_k"],
        "exposure_scale": dev["exposure_scale"],
        "color_scale": dev["color_scale"],
        "detail_scale": dev["detail_scale"],
        "hue_bias": dev["hue_bias"],
        "recenter": params["recenter"] if params.get("recenter_override") else dev["recenter"],
        "max_chroma": params["max_chroma"] if params.get("max_chroma_override") else dev["max_chroma"],
        "chroma_plane": params["chroma_plane"] if params.get("chroma_plane_override") else dev["chroma_plane"],
    }
