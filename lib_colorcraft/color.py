import torch

def _color_adjust(denoised, t, anchor):
    anchor = anchor.to(dtype=denoised.dtype, device=denoised.device)
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    x = denoised - anchor
    reduce_dims = tuple(range(2, denoised.dim()))
    mx = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    xn = x / mx
    signs = torch.sign(xn)
    ax = xn.abs()
    if t > 0:
        y = 1 - (1 - ax).pow(1.0 / (t + 1))
    else:
        y = 1 - (1 - ax).pow(abs(t) + 1)
    return anchor + y * mx * signs


def _color_adjust_legacy(denoised, t, anchor):
    anchor = anchor.to(dtype=denoised.dtype, device=denoised.device)
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    return torch.lerp(denoised, anchor, t)


def apply_contrast(x, alpha, neutral_anchor):
    if alpha == 0:
        return x
    return _color_adjust(x, -alpha, neutral_anchor)


def apply_color_shift(x, alpha, mode, color_anchor):
    if alpha == 0:
        return x
    fn = _color_adjust_legacy if mode == "legacy" else _color_adjust
    return fn(x, alpha, color_anchor)


def build_color_latent(vae, device, red, green, blue, brightness):
    """Encodes a flat-color 512x512 image and averages spatial (and
    temporal, if present) dims to get one anchor value per channel.
    device is host-specific, supplied by the caller."""
    img = torch.full((1, 512, 512, 3), 0.5, device=device)
    img[..., 0] += red
    img[..., 1] += green
    img[..., 2] += blue
    img += brightness
    latent = vae.encode(img)
    return latent.mean(dim=tuple(range(2, latent.dim())))[0]


def to_model_space(vae, latent_format, anchor):
    if latent_format is None or not hasattr(latent_format, "process_in"):
        return anchor
    dims = getattr(vae, "latent_dim", None)
    if dims is None:
        dims = getattr(latent_format, "latent_dimensions", 2)
    shaped = anchor.view((1, anchor.shape[0]) + (1,) * dims)
    return latent_format.process_in(shaped)[0].reshape(-1)