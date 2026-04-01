#!/usr/bin/env python3
"""
Advanced Image Processing Assistant
Resizes images and overlays a company logo with production-quality output.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageChops

# ── Constants ────────────────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATHS = {
    "dark":  ASSETS_DIR / "logo_dark.png",
    "light": ASSETS_DIR / "logo_light.png",
}

ResizeMode   = Literal["contain", "cover", "stretch"]
LogoPosition = Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"]
BlendMode    = Literal["normal", "multiply", "overlay"]
OutputFormat = Literal["png", "jpeg", "webp"]

# Candidate positions evaluated during auto-placement (corners first, center last)
_AUTO_CANDIDATES: list[LogoPosition] = [
    "bottom-right", "bottom-left", "top-right", "top-left", "center"
]


# ── Auto placement ───────────────────────────────────────────────────────────

def _region_coords(
    position: LogoPosition,
    canvas_w: int, canvas_h: int,
    logo_w: int, logo_h: int,
    margin: int,
) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the logo footprint for a given position."""
    positions = {
        "top-left":     (margin,                   margin),
        "top-right":    (canvas_w - logo_w - margin, margin),
        "bottom-left":  (margin,                   canvas_h - logo_h - margin),
        "bottom-right": (canvas_w - logo_w - margin, canvas_h - logo_h - margin),
        "center":       ((canvas_w - logo_w) // 2,  (canvas_h - logo_h) // 2),
    }
    x, y = positions[position]
    return x, y, x + logo_w, y + logo_h


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Fast Sobel edge magnitude on a float32 grayscale array (H×W, 0–255)."""
    # Horizontal / vertical kernels via slicing (no scipy needed)
    gx = (gray[1:-1, 2:].astype(np.float32) - gray[1:-1, :-2].astype(np.float32))
    gy = (gray[2:, 1:-1].astype(np.float32) - gray[:-2, 1:-1].astype(np.float32))
    return np.sqrt(gx ** 2 + gy ** 2)


def auto_place_logo(
    canvas: Image.Image,
    logo_w: int,
    logo_h: int,
    margin: int,
) -> Tuple[LogoPosition, Literal["dark", "light"]]:
    """
    Analyse the canvas and return the best (position, logo_variant).

    Scoring per candidate region (higher = better placement):
      - Edge density  : fewer edges → calmer background → logo won't clash
      - Color variance: low variance → uniform area → logo is readable
    Weights: 60% edge, 40% variance.

    Logo variant is chosen from the winning region's mean brightness:
      - dark background  (mean < 110) → use light logo
      - light background (mean > 145) → use dark logo
      - mid-tone                       → pick whichever has more contrast
    """
    cw, ch = canvas.size

    # Work on a small thumbnail for speed — 256px wide max
    scale = min(1.0, 256 / cw)
    tw, th = max(1, round(cw * scale)), max(1, round(ch * scale))
    thumb  = canvas.resize((tw, th), Image.BILINEAR).convert("RGB")
    gray   = np.array(thumb.convert("L"), dtype=np.float32)   # (H, W)
    edges  = _sobel_edges(gray)                                # (H-2, W-2)

    lw = max(1, round(logo_w * scale))
    lh = max(1, round(logo_h * scale))
    m  = max(0, round(margin * scale))

    best_pos   : LogoPosition = "bottom-right"
    best_score : float        = -1.0
    best_mean  : float        = 128.0

    for pos in _AUTO_CANDIDATES:
        x0, y0, x1, y1 = _region_coords(pos, tw, th, lw, lh, m)

        # Guard: skip if region falls outside thumbnail
        if x0 < 0 or y0 < 0 or x1 > tw or y1 > th:
            continue

        # Clamp edge array bounds (edges is 2px smaller on each side)
        ex0, ey0 = max(0, x0 - 1), max(0, y0 - 1)
        ex1, ey1 = min(edges.shape[1], x1 - 1), min(edges.shape[0], y1 - 1)
        if ex1 <= ex0 or ey1 <= ey0:
            continue

        region_gray  = gray[y0:y1, x0:x1]
        region_edges = edges[ey0:ey1, ex0:ex1]

        mean_edge    = float(region_edges.mean())
        color_std    = float(region_gray.std())
        region_mean  = float(region_gray.mean())

        # Normalize: Sobel max ≈ 360, std max ≈ 128
        edge_score     = 1.0 - min(mean_edge / 360.0, 1.0)
        variance_score = 1.0 - min(color_std  / 128.0, 1.0)
        score          = 0.60 * edge_score + 0.40 * variance_score

        if score > best_score:
            best_score = score
            best_pos   = pos
            best_mean  = region_mean

    # Choose variant based on brightness of winning region
    if best_mean < 110:
        variant: Literal["dark", "light"] = "light"   # dark bg → light logo
    elif best_mean > 145:
        variant = "dark"                               # light bg → dark logo
    else:
        # Mid-tone: pick whichever logo has higher contrast
        variant = "light" if best_mean < 128 else "dark"

    return best_pos, variant


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_image(
    img: Image.Image,
    width: int,
    height: int,
    mode: ResizeMode,
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """Return a new image at (width × height) using the specified resize mode."""

    src_w, src_h = img.size
    target = (width, height)

    if mode == "stretch":
        return img.convert("RGBA").resize(target, Image.LANCZOS)

    src_ratio = src_w / src_h
    tgt_ratio = width / height

    if mode == "contain":
        if src_ratio > tgt_ratio:          # wider than target → fit width
            new_w = width
            new_h = round(width / src_ratio)
        else:                               # taller than target → fit height
            new_h = height
            new_w = round(height * src_ratio)

        resized = img.convert("RGBA").resize((new_w, new_h), Image.LANCZOS)
        canvas  = Image.new("RGBA", target, background_color)
        offset  = ((width - new_w) // 2, (height - new_h) // 2)
        canvas.paste(resized, offset, resized)
        return canvas

    # mode == "cover"
    if src_ratio > tgt_ratio:              # wider → fit height, crop sides
        new_h = height
        new_w = round(height * src_ratio)
    else:                                  # taller → fit width, crop top/bottom
        new_w = width
        new_h = round(width / src_ratio)

    resized = img.convert("RGBA").resize((new_w, new_h), Image.LANCZOS)
    left    = (new_w - width)  // 2
    top     = (new_h - height) // 2
    return resized.crop((left, top, left + width, top + height))


# ── Logo overlay ──────────────────────────────────────────────────────────────

def _apply_opacity(logo: Image.Image, opacity: float) -> Image.Image:
    """Scale the alpha channel of a logo by opacity (0.0–1.0)."""
    if opacity >= 1.0:
        return logo
    r, g, b, a = logo.split()
    a = a.point(lambda x: round(x * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def _blend(base: Image.Image, logo: Image.Image, blend_mode: BlendMode) -> Image.Image:
    """Composite logo onto base using the requested blend mode."""
    if blend_mode == "normal":
        base.paste(logo, (0, 0), logo)
        return base

    # Extract regions
    logo_arr  = np.array(logo,                    dtype=np.float32) / 255.0
    base_crop = base.crop((0, 0, logo.width, logo.height))
    base_arr  = np.array(base_crop.convert("RGBA"), dtype=np.float32) / 255.0

    rgb_logo = logo_arr[..., :3]
    rgb_base = base_arr[..., :3]
    alpha    = logo_arr[..., 3:4]          # (H, W, 1) — logo's final alpha

    if blend_mode == "multiply":
        blended_rgb = rgb_base * rgb_logo
    elif blend_mode == "overlay":
        low  = 2.0 * rgb_base * rgb_logo
        high = 1.0 - 2.0 * (1.0 - rgb_base) * (1.0 - rgb_logo)
        blended_rgb = np.where(rgb_base < 0.5, low, high)
    else:
        blended_rgb = rgb_logo

    # Alpha-composite blended logo over base
    out_rgb  = alpha * blended_rgb + (1.0 - alpha) * rgb_base
    out_rgba = np.concatenate([out_rgb, base_arr[..., 3:4]], axis=-1)
    out_rgba = (np.clip(out_rgba, 0.0, 1.0) * 255).astype(np.uint8)

    blended_patch = Image.fromarray(out_rgba, "RGBA")
    base.paste(blended_patch, (0, 0), blended_patch)
    return base


def overlay_logo(
    canvas: Image.Image,
    logo_variant: Literal["dark", "light"],
    position: LogoPosition,
    scale_pct: float,
    margin: int,
    opacity: float,
    blend_mode: BlendMode = "normal",
) -> Image.Image:
    """Overlay the pre-loaded logo onto canvas and return the composited image."""

    logo_path = LOGO_PATHS[logo_variant]
    if not logo_path.exists():
        raise FileNotFoundError(f"Logo not found: {logo_path}")

    logo = Image.open(logo_path).convert("RGBA")

    # Scale logo proportionally
    logo_w = max(1, round(canvas.width * scale_pct / 100))
    ratio  = logo_w / logo.width
    logo_h = max(1, round(logo.height * ratio))
    logo   = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # Clamp so logo never exceeds canvas
    logo_w = min(logo_w, canvas.width  - 2 * margin)
    logo_h = min(logo_h, canvas.height - 2 * margin)
    if (logo_w, logo_h) != logo.size:
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # Apply opacity
    logo = _apply_opacity(logo, opacity)

    # Calculate top-left paste coordinates
    cw, ch = canvas.size
    positions = {
        "top-left":     (margin,               margin),
        "top-right":    (cw - logo_w - margin, margin),
        "bottom-left":  (margin,               ch - logo_h - margin),
        "bottom-right": (cw - logo_w - margin, ch - logo_h - margin),
        "center":       ((cw - logo_w) // 2,   (ch - logo_h) // 2),
    }
    px, py = positions[position]

    result = canvas.copy()

    if blend_mode == "normal":
        result.paste(logo, (px, py), logo)
    else:
        # Crop the region, blend, paste back
        region = result.crop((px, py, px + logo_w, py + logo_h)).convert("RGBA")
        blended = _blend(region, logo, blend_mode)
        result.paste(blended, (px, py), blended)

    return result


# ── Save ──────────────────────────────────────────────────────────────────────

def save_image(
    img: Image.Image,
    output_path: Path,
    fmt: OutputFormat,
    quality: int = 95,
) -> None:
    fmt = fmt.lower()
    if fmt == "jpeg":
        img = img.convert("RGB")          # JPEG has no alpha
        img.save(output_path, format="JPEG", quality=quality, subsampling=0)
    elif fmt == "webp":
        img.save(output_path, format="WEBP", quality=quality, lossless=(quality == 100))
    else:
        img.save(output_path, format="PNG", optimize=True)


# ── Public API ────────────────────────────────────────────────────────────────

def process_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    resize_mode: ResizeMode       = "cover",
    logo_variant: str             = "auto",
    logo_position: str            = "auto",
    logo_scale: float             = 15.0,
    logo_margin: int              = 20,
    logo_opacity: float           = 0.85,
    logo_blend_mode: BlendMode    = "normal",
    background_color: str         = "white",
    output_format: OutputFormat   = "png",
    quality: int                  = 95,
) -> dict:
    """
    Full pipeline: load → resize → overlay logo → save.

    logo_position and logo_variant both accept "auto" (default) to let the
    system pick the best placement and variant from the image content.

    Returns a metadata dict describing what was done.
    """

    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    valid_variants = list(LOGO_PATHS) + ["auto"]
    if logo_variant not in valid_variants:
        raise ValueError(f"logo_variant must be one of {valid_variants}, got '{logo_variant}'")

    # Parse background color
    bg = _parse_color(background_color)

    # Load & resize first so auto-placement analyses the final canvas
    img = Image.open(input_path)
    original_size = img.size
    resized = resize_image(img, width, height, resize_mode, bg)

    # ── Resolve "auto" position / variant ────────────────────────────────────
    auto_used = logo_position == "auto" or logo_variant == "auto"

    if auto_used:
        # Compute logo size at the requested scale so the analyser gets the
        # correct footprint (aspect ratio of the chosen/auto logo).
        # If variant is also auto we use the light logo dimensions (same ratio).
        probe_variant = "light" if logo_variant == "auto" else logo_variant
        probe_logo    = Image.open(LOGO_PATHS[probe_variant])
        lw = max(1, round(width * logo_scale / 100))
        lh = max(1, round(lw * probe_logo.height / probe_logo.width))

        auto_pos, auto_var = auto_place_logo(resized, lw, lh, logo_margin)

        if logo_position == "auto":
            logo_position = auto_pos
        if logo_variant == "auto":
            logo_variant = auto_var

    # ── Logo overlay ──────────────────────────────────────────────────────────
    result = overlay_logo(
        resized,
        logo_variant   = logo_variant,
        position       = logo_position,
        scale_pct      = logo_scale,
        margin         = logo_margin,
        opacity        = logo_opacity,
        blend_mode     = logo_blend_mode,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, output_path, output_format, quality)

    logo_px    = round(width * logo_scale / 100)
    logo_ratio = Image.open(LOGO_PATHS[logo_variant]).size
    logo_h_px  = round(logo_px * logo_ratio[1] / logo_ratio[0])

    meta = {
        "input":          str(input_path),
        "output":         str(output_path),
        "original_size":  f"{original_size[0]}×{original_size[1]}",
        "final_size":     f"{width}×{height}",
        "resize_mode":    resize_mode,
        "logo_variant":   logo_variant + (" (auto)" if auto_used else ""),
        "logo_position":  logo_position + (" (auto)" if auto_used else ""),
        "logo_scale":     f"{logo_scale}%  →  {logo_px}×{logo_h_px}px",
        "logo_margin":    f"{logo_margin}px",
        "logo_opacity":   logo_opacity,
        "logo_blend":     logo_blend_mode,
        "output_format":  output_format.upper(),
        "quality":        quality if output_format != "png" else "lossless",
    }
    # Expose resolved values for the frontend to reflect
    if auto_used:
        meta["auto_position"] = logo_position
        meta["auto_variant"]  = logo_variant
    return meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_color(color: str) -> Tuple[int, int, int, int]:
    color = color.strip().lower()
    named = {
        "white":       (255, 255, 255, 255),
        "black":       (0,   0,   0,   255),
        "transparent": (0,   0,   0,   0),
        "gray":        (128, 128, 128, 255),
        "grey":        (128, 128, 128, 255),
    }
    if color in named:
        return named[color]
    if color.startswith("#"):
        h = color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (r, g, b, 255)
        if len(h) == 8:
            r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
            return (r, g, b, a)
    raise ValueError(f"Unrecognized background_color: '{color}'. Use a name or #RRGGBB hex.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resize an image and overlay the company logo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  help="Path to the input image")
    p.add_argument("output", help="Path for the processed output image")

    # Dimensions
    p.add_argument("-W", "--width",  type=int, required=True,  help="Output width in pixels")
    p.add_argument("-H", "--height", type=int, required=True,  help="Output height in pixels")
    p.add_argument("--resize-mode", choices=["contain", "cover", "stretch"],
                   default="cover", help="Resize strategy")

    # Logo
    p.add_argument("--logo",          choices=["dark", "light"], default="light",
                   help="Logo variant to use")
    p.add_argument("--logo-position",
                   choices=["top-left","top-right","bottom-left","bottom-right","center"],
                   default="bottom-right")
    p.add_argument("--logo-scale",    type=float, default=15.0,
                   help="Logo width as %% of image width")
    p.add_argument("--logo-margin",   type=int,   default=20,
                   help="Margin from edges in pixels")
    p.add_argument("--logo-opacity",  type=float, default=0.85,
                   help="Logo opacity 0.0–1.0")
    p.add_argument("--logo-blend",    choices=["normal","multiply","overlay"],
                   default="normal")

    # Background / output
    p.add_argument("--bg",      default="white",
                   help="Background color for 'contain' mode (name or #RRGGBB)")
    p.add_argument("--format",  choices=["png","jpeg","webp"], default="png")
    p.add_argument("--quality", type=int, default=95,
                   help="Compression quality for JPEG/WebP (1–100)")
    p.add_argument("--json",    action="store_true",
                   help="Print metadata as JSON")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        meta = process_image(
            input_path      = args.input,
            output_path     = args.output,
            width           = args.width,
            height          = args.height,
            resize_mode     = args.resize_mode,
            logo_variant    = args.logo,
            logo_position   = args.logo_position,
            logo_scale      = args.logo_scale,
            logo_margin     = args.logo_margin,
            logo_opacity    = args.logo_opacity,
            logo_blend_mode = args.logo_blend,
            background_color= args.bg,
            output_format   = args.format,
            quality         = args.quality,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(meta, indent=2))
    else:
        w = max(len(k) for k in meta)
        print("\n✓ Image processed successfully\n")
        for k, v in meta.items():
            print(f"  {k:<{w}}  {v}")
        print()


if __name__ == "__main__":
    main()
