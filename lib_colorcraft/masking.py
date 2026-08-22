import torch
import torch.nn.functional as F

from .vectors import chroma_axes, _flatten, _unflatten

MASK_AXIS_OPTIONS = [
    "exposure", "hue", "saturation", "temperature", "tint",
    "temp+tint", "temp-tint", "lab-a", "lab-b", "lab-a+b", "lab-a-b",
    "clarity", "sharpness",
]


def axis_scale_for(axis, dev):
    """Which of dev's normalization scales applies to a real axis --
    shared by resolve_mask_tensor and the Forge Debug panel so the two
    never drift apart. Not meaningful for hue/saturation pseudo-axes,
    which have their own mapping; callers branch on those first."""
    if axis == "exposure":
        return dev["exposure_scale"]
    if axis in ("clarity", "sharpness"):
        return dev["detail_scale"]
    return dev["color_scale"]


# Every axis is normalized to roughly +-1 before reaching _mask_shape.
# HARDNESS_GAIN scales the hardness widget to match; width=2 always means
# "the whole domain", for every axis type including hue.
HARDNESS_GAIN = 5.0


def _mask_shape(vals, mode, center, hardness, width, strength=1.0):
    """strength lerps toward all-ones (1.0=mask as computed, 0.0=fully
    disabled). Skipped for split, whose output already spans [-1,1] as a
    signed direction, not a [0,1] gate."""
    c, s, w = center, hardness * HARDNESS_GAIN, width
    if mode == "highs":
        mask = torch.sigmoid((vals - c) * s)
    elif mode == "lows":
        mask = torch.sigmoid(-(vals - c) * s)
    elif mode == "split":
        return torch.tanh((vals - c) * s)
    elif mode in ("range", "protect range"):
        excess = ((vals - c).abs() - w / 2.0).clamp_min(0.0)
        g = torch.exp(-0.5 * (excess * s) ** 2)
        mask = (1.0 - g) if mode == "protect range" else g
    else:
        mask = torch.ones_like(vals)
    return 1.0 + strength * (mask - 1.0)


def compute_projection(x, basis, scale=1.0):
    """Raw per-pixel projection of x onto basis, normalized by scale --
    the value _mask_shape receives before any shaping. Shared by
    compute_mask and the Forge Debug panel's axis-projection view."""
    basis = basis / basis.norm()
    z, shape = _flatten(x)
    vals = (z @ basis) / scale
    B, C, H, W = shape
    return vals.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B,1,H,W] -- broadcasts over channels


def compute_mask(x, mask_basis, mode, center, hardness, width=0.0, strength=1.0, scale=1.0):
    """scale divides the raw projection before shaping, normalizing each
    real axis's range to roughly +-1 on any model."""
    proj = compute_projection(x, mask_basis, scale)
    return _mask_shape(proj, mode, center, hardness, width, strength)


def compute_hue_projection(x, axis1_basis, axis2_basis, hue_bias=0.0):
    """Raw per-pixel hue angle in the chroma-plane pair, normalized to
    -1..1 -- angle = atan2(c2,c1) - hue_bias, wrapped to (-pi,pi], divided
    by pi. This is the raw, unmodulated angle the mask gate itself uses,
    with no magnitude/saturation weighting."""
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    angle = torch.atan2(c2, c1) - hue_bias
    pi = torch.pi
    wrapped = (angle + pi) % (2 * pi) - pi
    B, C, H, W = shape
    return (wrapped / pi).reshape(B, H, W, 1).permute(0, 3, 1, 2)


def compute_hue_mask(x, axis1_basis, axis2_basis, mode, center, hardness, width=0.0, strength=1.0, hue_bias=0.0):
    """Circular counterpart to compute_mask: gates by hue angle (atan2 of
    the two chroma-plane projections) instead of a linear axis, for
    isolating an angular range like skin tones. center is in raw radians;
    the wrapped angular difference is normalized by pi before reaching
    _mask_shape. hue_bias (radians, per-model) rotates the angle so
    mask_center=0 lines up across models."""
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    angle = torch.atan2(c2, c1) - hue_bias
    pi = torch.pi
    diff = (angle - center * pi + pi) % (2 * pi) - pi  # wrapped difference, range (-pi, pi]
    mask = _mask_shape(diff / pi, mode, 0.0, hardness, width, strength)
    B, C, H, W = shape
    return mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)


def compute_saturation_projection(x, axis1_basis, axis2_basis, r_max=2.2):
    """Raw chroma magnitude r, mapped r=0 -> -1 and r=r_max -> +1 (clamped
    beyond r_max). Shared by compute_saturation_mask and the Forge Debug
    panel's saturation projection."""
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)
    if r_max == 0:
        mapped = torch.full_like(r, -1.0)
    else:
        mapped = ((r / r_max) * 2.0 - 1.0).clamp(-1.0, 1.0)
    B, C, H, W = shape
    return mapped.reshape(B, H, W, 1).permute(0, 3, 1, 2)


def compute_saturation_mask(x, axis1_basis, axis2_basis, mode, center, hardness, width=0.0, r_max=2.2, strength=1.0):
    """Pseudo-axis for chroma magnitude r (always >= 0). Maps r=0 -> -1,
    r=r_max -> +1, clamped beyond r_max, matching the +-1 convention every
    other axis uses."""
    mapped = compute_saturation_projection(x, axis1_basis, axis2_basis, r_max)
    return _mask_shape(mapped, mode, center, hardness, width, strength)


def gaussian_blur_mask(mask, sigma):
    """Spatially blurs an already-resolved [B,1,H,W] mask so its influence
    spreads past the exact pixels that satisfied its gate condition.
    Separable (two 1D passes) with reflect padding so strength doesn't
    fade at the latent's edges. sigma is in latent pixels."""
    if sigma <= 0:
        return mask
    radius = max(1, int(round(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    mask = F.pad(mask, (radius, radius, 0, 0), mode="reflect")
    mask = F.conv2d(mask, kernel.view(1, 1, 1, -1))
    mask = F.pad(mask, (0, 0, radius, radius), mode="reflect")
    mask = F.conv2d(mask, kernel.view(1, 1, -1, 1))
    return mask


def apply_mask_spread(mask, spread):
    """Compensates for the coverage a blur dilutes -- a gamma curve (like a
    compositing choke/spread on a blurred matte) so the mask's covered area
    actually grows or shrinks, rather than just washing brighter/darker.
    spread=0 is identity; positive grows the mask, negative shrinks it.
    Clamped to 0..1 first to avoid NaN from pow() on floating-point noise."""
    if spread == 0:
        return mask
    gamma = 2.0 ** (-spread)
    return torch.clamp(mask, 0.0, 1.0) ** gamma


def apply_mask_normalize(mask):
    """Per-batch-item min-max stretch to exactly [0,1]. A mask's actual
    values after blur/spread can span a much narrower range than its
    [0,1] domain (e.g. [0.4, 0.6]) -- apply_mask_contrast's ease curve is
    calibrated around the full domain, so pushing contrast on a
    narrow-range mask produces a weaker, less predictable result than
    expected. Normalizing first means contrast always sees a genuinely
    full-range input. A uniform mask (min==max) stays uniform rather
    than dividing by zero."""
    B = mask.shape[0]
    flat = mask.reshape(B, -1)
    mn = flat.amin(dim=1).view(B, 1, 1, 1)
    mx = flat.amax(dim=1).view(B, 1, 1, 1)
    return (mask - mn) / (mx - mn).clamp_min(1e-8)


def apply_mask_contrast(mask, contrast):
    """Photoshop's "Contrast" mask-refinement slider: steepens (or
    flattens) the transition around the mask's fixed midpoint (0.5) --
    values above push further toward 1, values below toward 0; 0, 0.5,
    and 1 stay fixed. contrast=0 is identity. Meant to run after spread,
    not before: spread reshapes where the transition sits, contrast then
    crisps up the result."""
    if contrast == 0:
        return mask
    t = -contrast
    m = torch.clamp(mask, 0.0, 1.0)
    d = m - 0.5
    pos = d.clamp_min(0.0)
    neg = (-d).clamp_min(0.0)

    def ease(u):
        if t > 0:
            return 1 - (1 - u).pow(1.0 / (t + 1))
        return 1 - (1 - u).pow(abs(t) + 1)

    pos_y = ease((pos / 0.5).clamp(0.0, 1.0))
    neg_y = ease((neg / 0.5).clamp(0.0, 1.0))
    return (0.5 + pos_y * 0.5 - neg_y * 0.5).clamp(0.0, 1.0)


def resolve_mask_tensor(mask_spec, pre, cur_basis, dev, vae_downscale_factor):
    """Recursively resolves a COLORCRAFT_MASK spec into a per-pixel mask
    tensor. A spec is a leaf (mode/axis/center/hardness/width/strength
    dict), a combine node ({"operation","a","b"}), or a blur node
    ({"blur","a"}) -- combine/blur nodes resolve their children first,
    then transform. Only happens here, at sampling time, since pre/
    cur_basis don't exist at graph-build time.

    Operations are fuzzy (product t-norm etc.), not hard boolean, since
    the gates are continuous 0..1 strengths -- reducing to normal boolean
    behavior at the 0/1 extremes automatically."""
    if "blur" in mask_spec:
        mask = resolve_mask_tensor(mask_spec["a"], pre, cur_basis, dev, vae_downscale_factor)
        mask = gaussian_blur_mask(mask, mask_spec["blur"] / vae_downscale_factor)
        mask = apply_mask_spread(mask, mask_spec["spread"])
        if mask_spec.get("normalize", False):
            mask = apply_mask_normalize(mask)
        return apply_mask_contrast(mask, mask_spec.get("contrast", 0.0))

    if "operation" in mask_spec:
        a = resolve_mask_tensor(mask_spec["a"], pre, cur_basis, dev, vae_downscale_factor)
        b = resolve_mask_tensor(mask_spec["b"], pre, cur_basis, dev, vae_downscale_factor)
        operation = mask_spec["operation"]
        if operation == "and":
            return a * b
        if operation == "or":
            return torch.clamp(a + b - a * b, 0.0, 1.0)
        if operation == "subtract":  # a but not b
            return torch.clamp(a - b, 0.0, 1.0)
        if operation == "xor":  # in one but not both
            return torch.clamp(a + b - 2 * a * b, 0.0, 1.0)
        return a  # unreachable given the combo's fixed option list

    axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
    axis = mask_spec["mask_axis"]
    shape_args = (mask_spec["mask_mode"], mask_spec["mask_center"], mask_spec["mask_hardness"], mask_spec["mask_width"])
    if axis == "hue":
        return compute_hue_mask(pre, axis1, axis2, *shape_args, mask_spec["mask_strength"], hue_bias=dev["hue_bias"])
    if axis == "saturation":
        return compute_saturation_mask(pre, axis1, axis2, *shape_args, r_max=dev["max_chroma"], strength=mask_spec["mask_strength"])
    if axis in cur_basis:
        return compute_mask(pre, cur_basis[axis], *shape_args, mask_spec["mask_strength"], scale=axis_scale_for(axis, dev))
    return torch.ones_like(pre[:, :1])  # unknown axis -- full pass-through, matches the old bare "return out" no-op


def apply_mask_gate(pre, out, mask_spec, dev, cur_basis, vae_downscale_factor):
    """dev is the consuming module's own resolved dev settings, not
    derived from mask_spec. mask_spec supplies only the mask shape.
    vae_downscale_factor is host/family-supplied (see basis.py)."""
    if cur_basis is None:
        return out
    mask = resolve_mask_tensor(mask_spec, pre, cur_basis, dev, vae_downscale_factor)
    return pre + mask * (out - pre)