"""Debug-panel rendering, shared between hosts -- pure PIL/torch/numpy,
no host-specific API calls. Decoding a latent to pixels and reading a
live UI/generation object stay in each host's own wrapper."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageColor

from .masking import _mask_shape, axis_scale_for, compute_projection, compute_saturation_projection, compute_hue_projection
from .vectors import chroma_axes

DEBUG_COMPOSITE_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
    "white": (255, 255, 255), "black": (0, 0, 0),
}
# For text/histogram color -- PIL recognizes these as fill= strings
# directly, no RGB translation needed.
DEBUG_OVERLAY_COLORS = ["white", "none", "red", "green", "blue", "yellow", "cyan", "magenta", "black"]
DEBUG_AXIS_STYLES = ["colormap", "greyscale"]


def build_debug_colormap_lut():
    """256x3 uint8 LUT for the axis-projection colormap style, via plain
    numpy interpolation across 3 fixed stops: 0.0 -> (10,0,178) blue,
    0.5 -> (230,0,0) red, 1.0 -> (255,255,0) yellow."""
    stops_x = [0.0, 0.5, 1.0]
    stops_rgb = [(10, 0, 178), (230, 0, 0), (255, 255, 0)]
    xs = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, stops_x, [c[ch] for c in stops_rgb])
    return lut


DEBUG_COLORMAP_LUT = build_debug_colormap_lut()


def build_debug_hue_lut():
    """256x3 uint8 LUT matching colorcraft.js's hueAt() gradient: HSL hue
    sweep raw = 220 - t*360 (t = normalized angle 0..1), wrapped mod 360,
    at saturation=100%/lightness=50%."""
    import colorsys
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0  # 0..1, corresponds to wrapped angle -pi..pi
        hue_deg = (220 - t * 360) % 360
        r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, 0.5, 1.0)
        lut[i] = [round(r * 255), round(g * 255), round(b * 255)]
    return lut


DEBUG_HUE_LUT = build_debug_hue_lut()


def render_hue_images(angle, width, height, label, overlay_color, curve_infos=None, chroma_frac=None):
    """[B,1,H,W] wrapped angle (-1..1, see compute_hue_projection) -> list
    of RGB PIL images via DEBUG_HUE_LUT, one per batch item. chroma_frac
    (also [B,1,H,W], 0..1), if given, lerps toward mid-grey by
    (1-chroma_frac) -- desaturating where the angle is least meaningful
    (atan2 of near-zero chroma is noisy). Without it, every pixel renders
    at full hue saturation -- the literal angle the mask gate uses.

    overlay_color is ignored -- always black, guaranteed contrast against
    the always-bright hue image."""
    t = angle.detach().float().cpu()
    idx = (((t[:, 0] + 1.0) / 2.0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    images = []
    for b in range(t.shape[0]):
        rgb = DEBUG_HUE_LUT[idx[b]].astype(np.float32)
        if chroma_frac is not None:
            frac = chroma_frac[b, 0].clamp(0.0, 1.0).numpy()[..., None]
            grey = np.full_like(rgb, 128.0)
            rgb = grey * (1.0 - frac) + rgb * frac
        # NEAREST not BILINEAR: hue is circular, blending RGB across a
        # wrap seam (e.g. 359deg next to 1deg) would cut through the
        # OPPOSITE color instead of the short way around.
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB").resize((width, height), Image.NEAREST)
        this_curves = None
        if curve_infos:
            this_curves = [dict(ci, projection=ci["projection"][b, 0]) for ci in curve_infos]
        annotate_debug_image(img, label if t.shape[0] == 1 else f"{label} [{b}]", t[b, 0], "black", this_curves, angle_domain=True)
        images.append(img)
    return images


def collect_leaves(spec):
    """Recursively collects every genuine leaf in a (possibly nested)
    mask spec -- unwraps blur wrappers ({"blur","spread","a"}) and
    descends both sides of a combine ({"operation","a","b"}). Forge's
    and Comfy's specs share this exact shape, so this works unchanged
    on either."""
    if "blur" in spec:
        return collect_leaves(spec["a"])
    if "operation" in spec:
        return collect_leaves(spec["a"]) + collect_leaves(spec["b"])
    return [spec]


def build_curve_infos(spec, x, cur_basis, dev):
    """Collects every leaf in spec (see collect_leaves) and returns a
    list of curve_info dicts (mode/center/hardness/width/strength/
    projection), skipping any axis with no valid projection. Shared by
    Forge's Debug panel and Comfy's ColorcraftDebug.

    Hue leaves use the raw angle for the histogram (comparable to the
    "hue" axis projection view directly); the real center is kept and a
    "wrap": True flag tells annotate_debug_image's curve-drawing to
    evaluate _mask_shape with wraparound at each point, since a linear
    axis's raw-projection-plus-separate-center approach breaks for a
    circular one."""
    curve_infos = []
    for leaf in collect_leaves(spec):
        axis = leaf.get("mask_axis")
        if axis == "hue":
            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
            angle = compute_hue_projection(x, axis1, axis2, dev.get("hue_bias", 0.0))
            curve_infos.append({
                "mode": leaf["mask_mode"], "center": leaf["mask_center"], "hardness": leaf["mask_hardness"],
                "width": leaf["mask_width"], "strength": leaf["mask_strength"], "projection": angle,
                "wrap": True,
            })
            continue
        proj = compute_axis_projection(axis, x, cur_basis, dev)
        if proj is None:
            continue
        curve_infos.append({
            "mode": leaf["mask_mode"], "center": leaf["mask_center"], "hardness": leaf["mask_hardness"],
            "width": leaf["mask_width"], "strength": leaf["mask_strength"], "projection": proj,
        })
    return curve_infos


def compute_axis_projection(axis, x, cur_basis, dev):
    """Raw (pre-shape) projection for an axis name -- saturation via its
    own magnitude mapping, real axes via the shared axis_scale_for()
    resolve_mask_tensor also uses. Returns None for hue: its value is a
    circular angle, not a plain scalar."""
    if axis == "hue":
        return None
    if axis == "saturation":
        axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
        return compute_saturation_projection(x, axis1, axis2, dev["max_chroma"])
    return compute_projection(x, cur_basis[axis], axis_scale_for(axis, dev))


def downscale_latent_for_storage(x, max_dim=64):
    """Downscales a latent to at most max_dim on its longer spatial side,
    then moves it to CPU. Batch dimension is preserved."""
    b, c, h, w = x.shape
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.detach().cpu()


def annotate_debug_image(img, label, raw_values, overlay_color, curve_infos=None, show_legend=False, angle_domain=False):
    """The label always draws (white text on translucent black),
    independent of overlay_color. Everything else (min/max text,
    histogram, curve) is skipped when overlay_color == "none".
    show_legend draws a colormap gradient strip along the bottom edge,
    also independent of overlay_color.

    curve_infos is None, or a list of curve_info dicts (mode/center/
    hardness/width/strength/projection). A single-item list draws one
    fully-opaque histogram+curve. A multi-item list (a combined mask's
    full chain of leaves) overlays each at reduced alpha, with min/max
    text spanning all of them combined."""
    w, h = img.size
    # Text sizes (and surrounding padding/margins/offsets) scale relative
    # to a 512px reference on the SHORTER side -- using w alone blows up
    # text/offsets on wide landscape images, since the limiting dimension
    # for how much overlay content fits is whichever side is shorter.
    # Line widths and X_MARGIN/legend height stay unscaled -- not text.
    scale = min(w, h) / 512.0
    label_font_size = round(20 * scale)
    label_pos = (8 * scale, 4 * scale)
    label_pad = 3 * scale
    minmax_font_size = round(16 * scale)
    minmax_pos = (8 * scale, 32 * scale)
    text_h = 56 * scale  # vertical space reserved for label+min/max before the plot itself begins
    tick_font_size = round(11 * scale)
    tick_notch = 5 * scale
    tick_label_gap = 2 * scale

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    bbox = od.textbbox(label_pos, label, font_size=label_font_size)
    od.rectangle([bbox[0] - label_pad, bbox[1] - label_pad, bbox[2] + label_pad, bbox[3] + label_pad], fill=(0, 0, 0, 128))
    od.text(label_pos, label, fill=(255, 255, 255, 255), font_size=label_font_size)

    X_MARGIN = 20  # px reserved on each side; -1 maps to X_MARGIN, +1 maps to w - X_MARGIN
    plot_w = w - 2 * X_MARGIN

    def value_x(v):
        return X_MARGIN + (v + 1.0) / 2.0 * plot_w

    if show_legend:
        legend_h = 10
        legend_row = DEBUG_COLORMAP_LUT[np.linspace(0, 255, plot_w).astype(np.uint8)]
        legend_arr = np.repeat(legend_row[np.newaxis, :, :], legend_h, axis=0)
        legend_img = Image.fromarray(legend_arr, mode="RGB").convert("RGBA")
        overlay.paste(legend_img, (X_MARGIN, h - legend_h))

    zero_y = value_y = None
    if overlay_color != "none":
        sources = [ci["projection"] for ci in curve_infos] if curve_infos else [raw_values]
        d_overlay = ImageDraw.Draw(img)
        all_vals = torch.cat([s.flatten().float() for s in sources])
        vmin, vmax = all_vals.min().item(), all_vals.max().item()
        d_overlay.text(minmax_pos, f"{vmin:.4f} to {vmax:.4f}", fill=overlay_color, font_size=minmax_font_size)

        plot_top, plot_bottom = text_h, h - 6
        plot_h = max(1, plot_bottom - plot_top)
        old_pos_frac = 0.75
        curve_scale = 0.6
        strip_h = (1.0 - old_pos_frac) * plot_h * curve_scale
        zero_y = plot_bottom - strip_h / 2.0

        def value_y(v):
            return zero_y - v * (strip_h / 2.0)

        bins = 256
        hists = [torch.histc(s.flatten().float(), bins=bins, min=-1.0, max=1.0) for s in sources]
        hist_max = max((h_.max().item() for h_ in hists), default=0.0) or 1.0
        hist_scale = 0.25 * curve_scale
        bar_cap = plot_h * 0.85 * hist_scale
        bar_alpha = 178 if len(sources) == 1 else 128  # 70% single-source (as before), 50% each when overlaying several
        bar_rgba = ImageColor.getrgb(overlay_color) + (bar_alpha,)
        for h_ in hists:
            for i in range(bins):
                lx = value_x(-1.0 + 2.0 * i / (bins - 1))
                ly = (h_[i].item() / hist_max) * bar_cap
                if ly > 0:
                    od.line([(lx, zero_y), (lx, zero_y - ly)], fill=bar_rgba, width=2)

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))
    if overlay_color == "none":
        return img

    if curve_infos:
        curve_alpha = 255  # curves always fully opaque -- only histogram bars carry reduced alpha
        curve_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(curve_overlay)
        xs = torch.linspace(-1.0, 1.0, 200)
        for ci in curve_infos:
            if ci.get("wrap"):
                # Circular axis (hue): center can't just pass through to
                # _mask_shape like a linear axis does -- evaluate each xs
                # point wrapped relative to center first (center=0 after
                # that, matching compute_hue_mask's own convention), so
                # the curve correctly continues across the domain edge
                # when center sits near +-1, rather than looking cut off.
                xs_eval = (xs - ci["center"] + 1.0) % 2.0 - 1.0
                ys = _mask_shape(xs_eval, ci["mode"], 0.0, ci["hardness"], ci["width"], ci["strength"])
            else:
                ys = _mask_shape(xs, ci["mode"], ci["center"], ci["hardness"], ci["width"], ci["strength"])
            pts = [(value_x(xs[i].item()), value_y(ys[i].item())) for i in range(len(xs))]
            cd.line(pts, fill=(255, 136, 0, curve_alpha), width=4)  # #f80 as explicit RGBA
        img.paste(Image.alpha_composite(img.convert("RGBA"), curve_overlay).convert("RGB"), (0, 0))

    d = ImageDraw.Draw(img)
    d.line([(value_x(-1.0), zero_y), (value_x(1.0), zero_y)], fill=overlay_color, width=2)

    tick_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    for tv in tick_values:
        tx = value_x(tv)
        d.line([(tx, zero_y - tick_notch), (tx, zero_y + tick_notch)], fill=overlay_color, width=1)
        if angle_domain:
            if tv == 0.0:
                label_str = "0"
            elif abs(tv) == 1.0:
                label_str = "-pi" if tv < 0 else "pi"
            else:
                label_str = f"{tv:+.2f}pi"
        else:
            label_str = f"{tv:+.2f}" if tv != 0.0 else "0"
        lbbox = d.textbbox((0, 0), label_str, font_size=tick_font_size)
        lw = lbbox[2] - lbbox[0]
        d.text((tx - lw / 2, zero_y + tick_notch + tick_label_gap), label_str, fill=overlay_color, font_size=tick_font_size)

    return img


def debug_tensor_to_images(tensor, width, height, is_projection, is_signed, label, overlay_color,
                            curve_infos=None, axis_style="greyscale"):
    """[B,1,H,W] -> list of RGB PIL Images, one per batch item, upscaled
    to the given size. Projections are always +-1-normalized; "split"
    mode masks can go negative -- is_signed controls whether the display
    remap is (v+1)/2 (centers on 50% grey) or a plain 0..1 clamp.
    axis_style="colormap" only applies to axis projections."""
    t = tensor.detach().float().cpu()
    disp = ((t + 1.0) / 2.0).clamp(0.0, 1.0) if (is_projection or is_signed) else t.clamp(0.0, 1.0)
    use_colormap = is_projection and axis_style == "colormap"
    images = []
    for b in range(t.shape[0]):
        if use_colormap:
            idx = (disp[b, 0].numpy() * 255).astype(np.uint8)
            rgb = DEBUG_COLORMAP_LUT[idx]
            img = Image.fromarray(rgb, mode="RGB")
        else:
            arr = (disp[b, 0].numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L").convert("RGB")
        img = img.resize((width, height), Image.BILINEAR)
        this_curves = None
        if curve_infos:
            this_curves = [dict(ci, projection=ci["projection"][b, 0]) for ci in curve_infos]
        annotate_debug_image(img, label if t.shape[0] == 1 else f"{label} [{b}]", t[b, 0], overlay_color,
                              this_curves, show_legend=use_colormap)
        images.append(img)
    return images


def composite_mask_images(mask_tensor, color_rgb, base_images, label, overlay_color, curve_infos=None):
    """Lerps each base image toward color_rgb where the mask is positive,
    and toward the inverted color where negative (e.g. "split" mode) --
    magnitude |mask| is the lerp parameter either way. base_images wraps
    by modulo if the mask's batch exceeds the number of base images."""
    if not base_images:
        return []
    raw = mask_tensor.detach().float()
    inv_color = tuple(255 - c for c in color_rgb)
    images = []
    for b in range(raw.shape[0]):
        base = base_images[b % len(base_images)].convert("RGB")
        w, h = base.size
        m = raw[b:b + 1, 0:1]
        m_up = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        alpha = m_up.abs().clamp(0.0, 1.0).cpu().numpy()[..., None]
        sign_mask = (m_up >= 0).cpu().numpy()[..., None]
        target = np.where(sign_mask, np.array(color_rgb, dtype=np.float32), np.array(inv_color, dtype=np.float32))
        base_arr = np.array(base).astype(np.float32)
        blended = base_arr * (1 - alpha) + target * alpha
        composited = Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")
        this_curves = None
        if curve_infos:
            this_curves = [dict(ci, projection=ci["projection"][b, 0]) for ci in curve_infos]
        annotate_debug_image(composited, label if raw.shape[0] == 1 else f"{label} [{b}]", raw[b, 0], overlay_color, this_curves)
        images.append(composited)
    return images