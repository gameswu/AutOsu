"""
Skin texture loader for osu! std mode.

Reads skin.ini and loads all relevant textures as BGRA numpy arrays.
Pre-bakes hitcircle textures per combo colour using the exact same
compositing pipeline as osr2mp4-core:
  1. Multiplicative colour tint on hitcircle  (add_color)
  2. alpha_composite overlay ON TOP of tinted hitcircle  (overlayhitcircle)
  3. Resize to circle diameter * overlay_scale  (1.05×)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── skin.ini parser ──────────────────────────────────────────────────────

@dataclass
class SkinIni:
    """Parsed values from skin.ini relevant to std rendering."""
    name: str = ""
    author: str = ""
    version: str = "latest"
    hit_circle_overlap: int = -2          # pixel overlap of combo number digits
    hit_circle_prefix: str = "default"    # font prefix for combo numbers
    slider_style: int = 2
    slider_border: Tuple[int, int, int] = (255, 255, 255)
    slider_track_override: Optional[Tuple[int, int, int]] = None
    allow_slider_ball_tint: bool = False
    cursor_rotate: bool = False
    cursor_expand: bool = False
    cursor_centre: bool = True
    hit_circle_overlay_above_number: bool = True
    slider_ball_flip: bool = False
    combo_colours: List[Tuple[int, int, int]] = field(default_factory=list)
    spinner_approach_circle_colour: Tuple[int, int, int] = (255, 255, 255)


def _parse_rgb(val: str) -> Tuple[int, int, int]:
    val = val.split("//")[0].strip()
    parts = [int(c.strip()) for c in val.split(",")]
    return (parts[0], parts[1], parts[2])


def parse_skin_ini(path: Path) -> SkinIni:
    """Parse a skin.ini file manually (supports duplicate keys like Combo1..N)."""
    ini = SkinIni()
    if not path.exists():
        return ini

    current_section = ""
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            current_section = m.group(1).lower()
            continue

        sep_idx = line.find(":")
        if sep_idx == -1:
            continue
        k = line[:sep_idx].strip()
        v = line[sep_idx + 1:].strip()
        if "//" in v:
            v = v.split("//")[0].strip()
        kl = k.lower()

        if current_section == "general":
            if kl == "name":
                ini.name = v
            elif kl == "author":
                ini.author = v
            elif kl == "version":
                ini.version = v
            elif kl == "sliderstyle":
                ini.slider_style = int(v)
            elif kl == "cursorrotate":
                ini.cursor_rotate = v == "1"
            elif kl == "cursorexpand":
                ini.cursor_expand = v == "1"
            elif kl == "cursorcentre":
                ini.cursor_centre = v == "1"
            elif kl in ("hitcircleoverlayabovenumer", "hitcircleoverlayabovenumber"):
                ini.hit_circle_overlay_above_number = v == "1"
            elif kl == "allowsliderballtint":
                ini.allow_slider_ball_tint = v == "1"
            elif kl == "sliderballflip":
                ini.slider_ball_flip = v == "1"
        elif current_section == "colours":
            if kl.startswith("combo"):
                ini.combo_colours.append(_parse_rgb(v))
            elif kl == "sliderborder":
                ini.slider_border = _parse_rgb(v)
            elif kl == "slidertrackoverride":
                ini.slider_track_override = _parse_rgb(v)
            elif kl == "spinnerapproachcircle":
                ini.spinner_approach_circle_colour = _parse_rgb(v)
        elif current_section == "fonts":
            if kl == "hitcircleprefix":
                ini.hit_circle_prefix = v
            elif kl == "hitcircleoverlap":
                ini.hit_circle_overlap = int(v)

    return ini


# ── Image loading helpers ────────────────────────────────────────────────

def _load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as BGRA numpy array.  Returns None if file missing."""
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _load_hd_or_sd(skin_dir: Path, stem: str, ext: str = ".png") -> Tuple[Optional[np.ndarray], bool]:
    """Try loading @2x first, then SD.  Returns (image, is_x2)."""
    hd_path = skin_dir / f"{stem}@2x{ext}"
    img = _load_image(hd_path)
    if img is not None:
        return img, True
    sd_path = skin_dir / f"{stem}{ext}"
    img = _load_image(sd_path)
    if img is not None:
        return img, False
    return None, False


# ── osr2mp4-compatible image operations ──────────────────────────────────

def _tint_image(img: np.ndarray, colour: Tuple[int, int, int]) -> np.ndarray:
    """Multiplicative tint (RGB) keeping alpha intact.

    Matches osr2mp4's imageproc.add_color:
      R_out = R_in * colour_R / 255
      G_out = G_in * colour_G / 255
      B_out = B_in * colour_B / 255
    Our images are BGRA, so B=ch0, G=ch1, R=ch2.
    """
    out = img.copy().astype(np.float32)
    out[:, :, 0] = out[:, :, 0] * colour[2] / 255.0   # B *= blue
    out[:, :, 1] = out[:, :, 1] * colour[1] / 255.0   # G *= green
    out[:, :, 2] = out[:, :, 2] * colour[0] / 255.0   # R *= red
    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_composite(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Porter-Duff 'over' composite: top on bottom, both BGRA.

    Matches PIL's Image.alpha_composite (used by osr2mp4 with channel=4).
    """
    assert bottom.shape == top.shape and bottom.shape[2] == 4

    b_rgb = bottom[:, :, :3].astype(np.float32)
    b_a = bottom[:, :, 3:4].astype(np.float32) / 255.0
    t_rgb = top[:, :, :3].astype(np.float32)
    t_a = top[:, :, 3:4].astype(np.float32) / 255.0

    out_a = t_a + b_a * (1.0 - t_a)
    safe_a = np.maximum(out_a, 1e-6)
    out_rgb = (t_rgb * t_a + b_rgb * b_a * (1.0 - t_a)) / safe_a

    result = np.zeros_like(bottom)
    result[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return result


def _resize_to_diameter(img: np.ndarray, diameter_px: int) -> np.ndarray:
    """Resize image so its width/height equals *diameter_px*."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img
    return cv2.resize(img, (diameter_px, diameter_px), interpolation=cv2.INTER_AREA)


def _center_paste(canvas: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    """Paste sprite centered on canvas using alpha compositing (channel=4).

    Matches osr2mp4's imageproc.add with channel=4.
    Both must be BGRA.
    """
    ch, cw = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    x = cw // 2 - sw // 2
    y = ch // 2 - sh // 2

    # Compute overlap region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + sw)
    y2 = min(ch, y + sh)
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return canvas

    region = canvas[y1:y2, x1:x2]
    src = sprite[sy1:sy2, sx1:sx2]
    canvas[y1:y2, x1:x2] = _alpha_composite(region, src)
    return canvas


# ── Main skin loader ─────────────────────────────────────────────────────

OVERLAY_SCALE = 1.05       # osr2mp4: overlay_scale in Circles.py
DEFAULT_SPRITE_SIZE = 128  # osr2mp4: default_size in Circles.py


@dataclass
class SkinTextures:
    """All loaded textures ready for rendering, scaled to target circle diameter."""
    ini: SkinIni

    # Raw (unscaled, untinted) textures + whether they were @2x
    hitcircle_raw: Optional[np.ndarray] = None
    hitcircle_is_x2: bool = False
    hitcircle_overlay_raw: Optional[np.ndarray] = None
    hitcircle_overlay_is_x2: bool = False
    approach_circle_raw: Optional[np.ndarray] = None
    approach_circle_is_x2: bool = False
    slider_ball_raw: Optional[np.ndarray] = None
    slider_ball_is_x2: bool = False
    slider_follow_circle_raw: Optional[np.ndarray] = None
    slider_follow_circle_is_x2: bool = False
    slider_reverse_arrow_raw: Optional[np.ndarray] = None
    slider_reverse_arrow_is_x2: bool = False
    slider_score_point_raw: Optional[np.ndarray] = None
    cursor_raw: Optional[np.ndarray] = None
    cursor_is_x2: bool = False
    cursor_trail_raw: Optional[np.ndarray] = None
    cursor_trail_is_x2: bool = False
    cursor_middle_raw: Optional[np.ndarray] = None

    # Pre-baked hitcircle+overlay per combo colour (osr2mp4 overlayhitcircle)
    # Key = combo_colour_index, value = BGRA image at final circle diameter
    baked_hitcircles: Dict[int, np.ndarray] = field(default_factory=dict)
    baked_slider_circles: Dict[int, np.ndarray] = field(default_factory=dict)

    # Tinted approach circles per combo colour (stored at raw resolution)
    tinted_approach_circles: Dict[int, np.ndarray] = field(default_factory=dict)
    # Approach circle base size at approach_ratio=1 (computed from raw 1x size * radius_scale)
    approach_circle_base_size: int = 0

    # Number sprites scaled to match circle size
    number_sprites: Dict[int, np.ndarray] = field(default_factory=dict)
    number_width: int = 0           # width of a single digit sprite (all same)
    number_overlap_px: int = 0      # HitCircleOverlap * scale

    # Rendering metrics
    circle_diameter_px: int = 0
    combo_colours: List[Tuple[int, int, int]] = field(default_factory=list)


def load_skin(
    skin_dir: str | Path,
    combo_colours: List[Tuple[int, int, int]] | None = None,
    circle_diameter_px: int = 64,
    cs: float = 4.0,
    playfield_scale: float = 0.8,
) -> SkinTextures:
    """
    Load an osu! skin from *skin_dir*.

    Parameters
    ----------
    skin_dir : path to the skin folder
    combo_colours : list of RGB tuples (from beatmap [Colours] or skin.ini)
    circle_diameter_px : target circle diameter in render pixels
    cs : circle size (used for exact osr2mp4-matching sprite scaling)
    playfield_scale : render_h / 768 (osr2mp4: settings.scale = h/768)
    """
    skin_dir = Path(skin_dir)
    ini = parse_skin_ini(skin_dir / "skin.ini")

    if combo_colours is None:
        combo_colours = ini.combo_colours
    if not combo_colours:
        combo_colours = [(255, 192, 0), (0, 202, 0), (18, 124, 255), (242, 24, 57)]

    tex = SkinTextures(ini=ini, circle_diameter_px=circle_diameter_px,
                       combo_colours=combo_colours)

    # Load raw textures (prefer @2x)
    tex.hitcircle_raw, tex.hitcircle_is_x2 = _load_hd_or_sd(skin_dir, "hitcircle")
    tex.hitcircle_overlay_raw, tex.hitcircle_overlay_is_x2 = _load_hd_or_sd(skin_dir, "hitcircleoverlay")
    tex.approach_circle_raw, tex.approach_circle_is_x2 = _load_hd_or_sd(skin_dir, "approachcircle")

    # Slider ball can be animated (sliderb0, sliderb1, ...) — load first frame
    tex.slider_ball_raw, tex.slider_ball_is_x2 = _load_hd_or_sd(skin_dir, "sliderb0")
    if tex.slider_ball_raw is None:
        tex.slider_ball_raw, tex.slider_ball_is_x2 = _load_hd_or_sd(skin_dir, "sliderb")

    tex.slider_follow_circle_raw, tex.slider_follow_circle_is_x2 = _load_hd_or_sd(skin_dir, "sliderfollowcircle")
    tex.slider_reverse_arrow_raw, tex.slider_reverse_arrow_is_x2 = _load_hd_or_sd(skin_dir, "reversearrow")
    tex.slider_score_point_raw, _ = _load_hd_or_sd(skin_dir, "sliderscorepoint")
    tex.cursor_raw, tex.cursor_is_x2 = _load_hd_or_sd(skin_dir, "cursor")
    tex.cursor_trail_raw, tex.cursor_trail_is_x2 = _load_hd_or_sd(skin_dir, "cursortrail")
    tex.cursor_middle_raw, _ = _load_hd_or_sd(skin_dir, "cursormiddle")

    # Number sprites
    prefix = ini.hit_circle_prefix
    number_sprites_raw: Dict[int, Tuple[np.ndarray, bool]] = {}
    for digit in range(10):
        img, is_x2 = _load_hd_or_sd(skin_dir, f"{prefix}-{digit}")
        if img is not None:
            number_sprites_raw[digit] = (img, is_x2)

    # Build pre-baked assets
    _build_assets(tex, combo_colours, cs, playfield_scale, number_sprites_raw)

    return tex


def _scale_sprite(img: np.ndarray, is_x2: bool, scale: float) -> np.ndarray:
    """Scale a sprite by *scale*, halving first if @2x.

    Matches osr2mp4's YImage: if x2, divide scale by 2; then resize.
    """
    if is_x2:
        scale = scale / 2.0

    h, w = img.shape[:2]
    new_h = max(2, int(h * scale))
    new_h += new_h % 2  # osr2mp4 forces even dimensions
    new_w = max(2, int(w * scale))
    new_w += new_w % 2
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _build_assets(
    tex: SkinTextures,
    combo_colours: List[Tuple[int, int, int]],
    cs: float,
    playfield_scale: float,
    number_sprites_raw: Dict[int, Tuple[np.ndarray, bool]],
):
    """Build all pre-baked assets matching osr2mp4-core's prepare_circle().

    Key formulas from osr2mp4-core/Circles.py:
      cs_osu = (54.4 - 4.48 * CS)
      radius_scale = cs_osu * overlay_scale * 2 / default_size
                   = cs_osu * 1.05 * 2 / 128

    This radius_scale is used for both circles and slider heads.
    """
    cs_osu = 54.4 - 4.48 * cs
    # osr2mp4 uses the same radius_scale for both circles and slider heads:
    #   radius_scale = cs_osu * overlay_scale * 2 / default_size
    # Slider heads differ only in sprite file (sliderstartcircle vs hitcircle),
    # which usually falls back to hitcircle anyway.
    radius_scale = cs_osu * OVERLAY_SCALE * 2 / DEFAULT_SPRITE_SIZE

    # --- Pre-bake hitcircle + overlay per combo colour ---
    # Matches osr2mp4's overlayhitcircle():
    #   1. Tint hitcircle with combo colour (add_color)
    #   2. Create canvas = max(circle, overlay) size
    #   3. Paste tinted circle centered
    #   4. alpha_composite overlay on top (channel=4)
    #   5. Resize by radius_scale

    for idx, colour in enumerate(combo_colours):
        # --- Circle version ---
        baked = _bake_hitcircle(
            tex.hitcircle_raw, tex.hitcircle_is_x2,
            tex.hitcircle_overlay_raw, tex.hitcircle_overlay_is_x2,
            colour, radius_scale * playfield_scale,
        )
        if baked is not None:
            tex.baked_hitcircles[idx] = baked

        # --- Slider head version ---
        # osr2mp4: tries sliderstartcircle.png (fallback hitcircle.png) and
        # sliderstartcircleoverlay.png (fallback hitcircleoverlay.png).
        # We use the same hitcircle textures since few skins have separate
        # sliderstart ones — just reference the same baked image.
        if baked is not None:
            tex.baked_slider_circles[idx] = baked

        # --- Tinted approach circle ---
        if tex.approach_circle_raw is not None:
            tinted_ac = _tint_image(tex.approach_circle_raw, colour)
            tex.tinted_approach_circles[idx] = tinted_ac

    # --- Approach circle base size ---
    # osr2mp4: approach circle is loaded at 1x via YImage, then resized by
    # s * radius_scale in prepare_approach(). At s=1 (converged), the base
    # size = ac_1x_height * radius_scale * playfield_scale.
    if tex.approach_circle_raw is not None:
        ac_scale = radius_scale * playfield_scale
        if tex.approach_circle_is_x2:
            ac_scale /= 2.0
        ac_h = tex.approach_circle_raw.shape[0]
        ac_base = max(2, int(ac_h * ac_scale))
        ac_base += ac_base % 2  # force even
        tex.approach_circle_base_size = ac_base

    # --- Number sprites ---
    # osr2mp4's prepare_hitcirclenumber():
    #   circle_radius = (54.4 - 4.48 * CS) * scale * 0.8
    #   number_scale = circle_radius * 2 / 128 * 1.05
    # Then each digit is loaded at that scale (via YImage).
    number_scale = cs_osu * playfield_scale * 0.8 * 2 / DEFAULT_SPRITE_SIZE * 1.05

    for digit, (raw, is_x2) in number_sprites_raw.items():
        scaled = _scale_sprite(raw, is_x2, number_scale)
        tex.number_sprites[digit] = scaled

    # All digit sprites should have the same width (osr2mp4 uses nwidth from first sprite)
    if tex.number_sprites:
        first = next(iter(tex.number_sprites.values()))
        tex.number_width = first.shape[1]
        # overlap in rendered pixels: osr2mp4 uses int(HitCircleOverlap * scale)
        # where scale is the number_scale (same as what was passed to YImage)
        tex.number_overlap_px = int(tex.ini.hit_circle_overlap * number_scale)

    # --- Circle diameter for renderer use ---
    # The final baked circle diameter is determined by the actual baked image size
    if tex.baked_hitcircles:
        first_baked = next(iter(tex.baked_hitcircles.values()))
        tex.circle_diameter_px = first_baked.shape[1]


def _bake_hitcircle(
    circle_raw: Optional[np.ndarray],
    circle_is_x2: bool,
    overlay_raw: Optional[np.ndarray],
    overlay_is_x2: bool,
    colour: Tuple[int, int, int],
    scale: float,
) -> Optional[np.ndarray]:
    """Bake a single hitcircle+overlay image for one combo colour.

    Matches osr2mp4's overlayhitcircle():
      1. color_circle = add_color(circle, color)
      2. canvas = max(circle, overlay) size
      3. paste circle centered on canvas
      4. alpha_composite overlay on top
      5. resize by scale
    """
    if circle_raw is None:
        return None

    # Scale circle to 1x (if @2x, halve)
    circle_scale = 0.5 if circle_is_x2 else 1.0
    if circle_scale != 1.0:
        h, w = circle_raw.shape[:2]
        circle_1x = cv2.resize(circle_raw, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    else:
        circle_1x = circle_raw.copy()

    # Tint
    tinted = _tint_image(circle_1x, colour)

    # Overlay at 1x
    if overlay_raw is not None:
        overlay_scale = 0.5 if overlay_is_x2 else 1.0
        if overlay_scale != 1.0:
            oh, ow = overlay_raw.shape[:2]
            overlay_1x = cv2.resize(overlay_raw, (ow // 2, oh // 2), interpolation=cv2.INTER_AREA)
        else:
            overlay_1x = overlay_raw.copy()
    else:
        overlay_1x = None

    # Create canvas at max(circle, overlay) size
    ch, cw = tinted.shape[:2]
    if overlay_1x is not None:
        oh, ow = overlay_1x.shape[:2]
        max_w = max(cw, ow)
        max_h = max(ch, oh)
    else:
        max_w, max_h = cw, ch

    canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
    # Paste tinted circle centered
    cx = max_w // 2 - cw // 2
    cy = max_h // 2 - ch // 2
    canvas[cy:cy + ch, cx:cx + cw] = tinted

    # Alpha composite overlay on top
    if overlay_1x is not None:
        _center_paste(canvas, overlay_1x)

    # Resize by final scale
    final_h = max(2, int(max_h * scale))
    final_h += final_h % 2
    final_w = max(2, int(max_w * scale))
    final_w += final_w % 2
    result = cv2.resize(canvas, (final_w, final_h), interpolation=cv2.INTER_AREA)

    return result
