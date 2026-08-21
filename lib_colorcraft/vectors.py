import torch

def _flatten(x):
    """[B,C,H,W] -> ([N,C], shape-info to restore)."""
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(-1, C), (B, C, H, W)


def _unflatten(z, shape):
    B, C, H, W = shape
    return z.reshape(B, H, W, C).permute(0, 3, 1, 2)


def apply_vector_offset(x, basis, alpha):
    """Adds `alpha` to the projected coefficient along `basis` and reconstructs."""
    if alpha == 0:
        return x
    z, shape = _flatten(x)
    coeffs = z @ basis
    delta = torch.full_like(coeffs, alpha)
    z2 = z + delta.unsqueeze(1) * basis
    return _unflatten(z2, shape)


def apply_vector_scale(x, basis, alpha):
    """Scales the projected coefficient along `basis` by (1 + alpha) and
    reconstructs. Normalizes `basis` to unit length first, since this op
    uses the vector twice (project, then reconstruct) and any magnitude
    baked into the stored vector would otherwise get squared instead of
    applied once."""
    if alpha == 0:
        return x
    basis = basis / basis.norm()
    z, shape = _flatten(x)
    coeffs = z @ basis
    target = coeffs * (1.0 + alpha)
    delta = target - coeffs
    z2 = z + delta.unsqueeze(1) * basis
    out = _unflatten(z2, shape)
    return out


def apply_vibrance(x, axis1_basis, axis2_basis, alpha, k=1.0, recenter=1.0, r_max=0.0):
    """Boosts/reduces chroma along the given axis pair. At r_max=0
    (Saturation), plain linear scaling by alpha. At r_max!=0 (Vibrance),
    gain fades to zero both near r=0 and near r=r_max, peaking around the
    middle -- protects near-neutral pixels (hue is numerically unreliable
    there) and already-vivid pixels alike. `k` controls how fast the
    high-end fade kicks in.

    `recenter` is a 0..1 blend: 1.0 fully corrects chroma drift but also
    removes real color casts the image had on purpose (e.g. a dusk photo's
    pink tint); lower values blend between no correction and full
    correction."""
    if alpha == 0:
        return x
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    if recenter != 0:
        orig_mean = x.mean(dim=(2, 3), keepdim=True)
    z, shape = _flatten(x)
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)

    if r_max == 0:
        delta1 = alpha * c1
        delta2 = alpha * c2
    else:
        # fraction of remaining chroma headroom to consume
        headroom_frac = torch.clamp(r / r_max, 0, 1)
        fade = headroom_frac ** (1 / max(k, 1e-6))
        fade = fade * fade * (3 - 2 * fade)
        protect = 1.0 - fade 

        low_frac = torch.clamp(r / max(r_max * 0.5, 1e-6), 0, 1)
        low_fade = low_frac * low_frac * (3 - 2 * low_frac)
        shape_mult = low_fade * protect

        if alpha >= 0:
            gain = (alpha * shape_mult).clamp(0.0, 1.0)
            target_r = r + gain * (r_max - r) 

        else:
            gain = (alpha * shape_mult).clamp(-1.0, 0.0)
            target_r = r * (1.0 + gain) 

        target_r = target_r.clamp_min(0.0)
        scale = target_r / r.clamp_min(1e-8)
        delta1 = c1 * scale - c1
        delta2 = c2 * scale - c2

    z2 = z + delta1.unsqueeze(1) * axis1_basis + delta2.unsqueeze(1) * axis2_basis
    out = _unflatten(z2, shape)

    if recenter != 0:
        new_mean = out.mean(dim=(2, 3), keepdim=True)
        out = out - recenter * (new_mean - orig_mean)
    return out


def apply_chroma_contrast(x, axis1_basis, axis2_basis, gamma, r_max=2.2, chroma_center=0.0, recenter=1.0):
    """Contrast on chroma magnitude, pivoted around a model-calibrated max
    chroma point: pixels above the pivot push further up, pixels below push
    down toward zero, each side eased independently (chroma is one-sided
    and bounded at 0, so it needs separate curves, not one shared midpoint).

    `gamma` is curve steepness: 0 linear, positive steepens, negative
    flattens toward the pivot. `chroma_center` is -1..1 (r=0 -> -1,
    r=r_max -> +1); 0 sits at r=r_max/2. `recenter` -- see apply_vibrance."""
    if gamma == 0:
        return x
    t = -gamma
    axis1_basis = axis1_basis / axis1_basis.norm()
    axis2_basis = axis2_basis / axis2_basis.norm()
    if recenter != 0:
        orig_mean = x.mean(dim=(2, 3), keepdim=True)
    z, shape = _flatten(x)
    B, C, H, W = shape
    c1 = z @ axis1_basis
    c2 = z @ axis2_basis
    r = torch.sqrt(c1 * c1 + c2 * c2)

    p = max(((chroma_center + 1.0) / 2.0) * r_max, 0.0)

    d = r - p
    pos = d.clamp_min(0.0)
    neg = (-d).clamp_min(0.0)
    pos_mx = pos.reshape(B, H * W).amax(dim=1, keepdim=True).clamp_min(1e-6).expand(-1, H * W).reshape(-1)
    neg_mx = max(p, 1e-6)  # room to fall is just the pivot itself, since r can't go below 0

    def ease(u):
        if t > 0:
            return 1 - (1 - u).pow(1.0 / (t + 1))
        return 1 - (1 - u).pow(abs(t) + 1)

    pos_y = ease((pos / pos_mx).clamp(0.0, 1.0))
    neg_y = ease((neg / neg_mx).clamp(0.0, 1.0))

    new_r = (p + pos_y * pos_mx - neg_y * neg_mx).clamp_min(0.0)
    ratio = new_r / r.clamp_min(1e-6)
    delta1 = (ratio - 1.0) * c1
    delta2 = (ratio - 1.0) * c2
    z2 = z + delta1.unsqueeze(1) * axis1_basis + delta2.unsqueeze(1) * axis2_basis
    out = _unflatten(z2, shape)
    if recenter != 0:
        new_mean = out.mean(dim=(2, 3), keepdim=True)
        out = out - recenter * (new_mean - orig_mean)
    return out


def apply_tone_compression(x, exposure_basis, ui_value):
    """UI range -1..1: 0 = no-op (backend factor 1), 1 = fully compressed to
    neutral (backend factor 0), -1 = expand tone (backend factor 2)."""
    if ui_value == 0:
        return x
    factor = 1.0 - ui_value
    alpha = factor - 1.0
    return apply_vector_scale(x, exposure_basis, alpha)



def chroma_axes(cur_basis, chroma_plane):
    return (
        (cur_basis["lab-a"], cur_basis["lab-b"]) if chroma_plane == "lab"
        else (cur_basis["temperature"], cur_basis["tint"])
    )