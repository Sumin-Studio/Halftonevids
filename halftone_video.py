#!/usr/bin/env python3
"""
Halftone Video Tool
===================
Convert any video into a halftone dot-pattern style.

Modes:
  bw    - Classic black & white halftone (dots on white background)
  color - CMYK-style color halftone (four-channel screen printing simulation)

Usage:
  python halftone_video.py input.mp4 output.mp4
  python halftone_video.py input.mp4 output.mp4 --mode color --cell-size 8
  python halftone_video.py input.mp4 output.mp4 --cell-size 15 --angle 30
  python halftone_video.py input.mp4 output.mp4 --invert
  python halftone_video.py input.mp4 output.mp4 --frames 30   # preview: first 30 frames
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_grid_points(w: int, h: int, cell_size: int, angle_deg: float) -> list[tuple[int, int]]:
    """
    Build a list of dot-center pixel coordinates for a halftone grid.

    The grid is square, aligned to `angle_deg`, and centred on the image.
    All points that fall within (or just outside) the image bounds are returned
    so every pixel gets covered.
    """
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    cx, cy = w / 2.0, h / 2.0
    diag = int(np.sqrt(w ** 2 + h ** 2))
    steps = diag // cell_size + 2          # generous margin

    points = []
    half = cell_size
    for row in range(-steps, steps + 1):
        for col in range(-steps, steps + 1):
            # Grid position in rotated space (centred on image)
            gx = col * cell_size
            gy = row * cell_size
            # Rotate back to image space
            ix = cos_a * gx - sin_a * gy + cx
            iy = sin_a * gx + cos_a * gy + cy
            ixi, iyi = int(round(ix)), int(round(iy))
            if -half <= ixi < w + half and -half <= iyi < h + half:
                points.append((ixi, iyi))

    return points


def sample_region(img_2d: np.ndarray, cx: int, cy: int, half: int) -> float:
    """Return mean value of a square region in a 2-D float array [0..1]."""
    h, w = img_2d.shape
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(img_2d[y1:y2, x1:x2]))


# ---------------------------------------------------------------------------
# Halftone modes
# ---------------------------------------------------------------------------

def apply_halftone_bw(
    frame: np.ndarray,
    cell_size: int = 10,
    angle: float = 45.0,
    invert: bool = False,
) -> np.ndarray:
    """
    Classic grayscale halftone: black dots on a white background.

    Dark pixels → large dots.  Bright pixels → small dots.
    Pass invert=True to flip this relationship.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    output = np.full((h, w, 3), 255, dtype=np.uint8)   # white background
    half = max(1, cell_size // 2)
    points = build_grid_points(w, h, cell_size, angle)

    for px, py in points:
        brightness = sample_region(gray, px, py, half)
        if invert:
            radius = int(half * brightness)
        else:
            radius = int(half * (1.0 - brightness))

        if radius > 0:
            draw_x = max(0, min(w - 1, px))
            draw_y = max(0, min(h - 1, py))
            cv2.circle(output, (draw_x, draw_y), radius, (0, 0, 0), -1, cv2.LINE_AA)

    return output


def apply_halftone_color(frame: np.ndarray, cell_size: int = 10) -> np.ndarray:
    """
    CMYK colour halftone (screen-printing simulation).

    Each ink channel (Cyan, Magenta, Yellow, Black) is rendered as dots on a
    rotated grid at its traditional angle.  The channels are composited with
    subtractive (multiply) blending, mimicking real ink on paper.

    Traditional screen angles:
        K (Black)   45°
        C (Cyan)    15°
        M (Magenta) 75°
        Y (Yellow)   0°
    """
    h, w = frame.shape[:2]

    # --- RGB → CMYK conversion -----------------------------------------------
    bgr = frame.astype(np.float32) / 255.0
    r, g, b = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]

    k_ch = 1.0 - np.maximum(np.maximum(r, g), b)
    denom = np.where(k_ch < 1.0, 1.0 - k_ch, 1.0)          # avoid /0
    c_ch = np.clip((1.0 - r - k_ch) / denom, 0.0, 1.0)
    m_ch = np.clip((1.0 - g - k_ch) / denom, 0.0, 1.0)
    y_ch = np.clip((1.0 - b - k_ch) / denom, 0.0, 1.0)

    # (name, channel_data, angle_deg)
    channels = [
        ("k", k_ch, 45.0),
        ("c", c_ch, 15.0),
        ("m", m_ch, 75.0),
        ("y", y_ch,  0.0),
    ]

    # Start with white (1.0 everywhere) — multiply blending
    output = np.ones((h, w, 3), dtype=np.float32)
    half = max(1, cell_size // 2)

    for name, ch_data, angle in channels:
        points = build_grid_points(w, h, cell_size, angle)

        # Rasterise dots into a float mask
        mask = np.zeros((h, w), dtype=np.float32)
        for px, py in points:
            intensity = sample_region(ch_data, px, py, half)
            radius = int(half * intensity)
            if radius > 0:
                draw_x = max(0, min(w - 1, px))
                draw_y = max(0, min(h - 1, py))
                cv2.circle(mask, (draw_x, draw_y), radius, 1.0, -1, cv2.LINE_AA)

        # Subtractive blending: each ink absorbs its complementary light
        # OpenCV stores channels as BGR: index 0=B, 1=G, 2=R
        absorb = 1.0 - mask
        if name == "c":
            output[:, :, 2] *= absorb          # Cyan   absorbs Red
        elif name == "m":
            output[:, :, 1] *= absorb          # Magenta absorbs Green
        elif name == "y":
            output[:, :, 0] *= absorb          # Yellow  absorbs Blue
        elif name == "k":
            output[:, :, 0] *= absorb          # Black   absorbs all
            output[:, :, 1] *= absorb
            output[:, :, 2] *= absorb

    return (output * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    input_path: Path,
    output_path: Path,
    mode: str = "bw",
    cell_size: int = 10,
    angle: float = 45.0,
    invert: bool = False,
    max_frames: int | None = None,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: cannot open '{input_path}'", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    # Try H.264 first, fall back to mp4v (works everywhere)
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            break
    else:
        print("Error: could not open video writer", file=sys.stderr)
        sys.exit(1)

    print(f"Input : {input_path}  ({width}x{height} @ {fps:.1f} fps)")
    print(f"Output: {output_path}")
    print(f"Mode  : {mode}  |  cell size: {cell_size} px  |  angle: {angle}°"
          + ("  |  inverted" if invert else ""))
    if max_frames:
        print(f"Frames: {max_frames} (preview)")
    print()

    processed = 0
    iterator = range(n_frames)
    if HAS_TQDM:
        iterator = tqdm(iterator, unit="frame", desc="Processing")

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "bw":
            result = apply_halftone_bw(frame, cell_size, angle, invert)
        else:
            result = apply_halftone_color(frame, cell_size)

        writer.write(result)
        processed += 1

    cap.release()
    writer.release()

    if not HAS_TQDM:
        print(f"Done — {processed} frames written.")
    else:
        print(f"\nDone — {processed} frames written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="halftone_video",
        description="Convert a video to halftone dot-pattern style.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Classic B&W halftone with 10-pixel cells:
  python halftone_video.py input.mp4 output.mp4

  # CMYK colour halftone with smaller cells (finer detail):
  python halftone_video.py input.mp4 output.mp4 --mode color --cell-size 8

  # B&W with a 30-degree grid angle and larger dots:
  python halftone_video.py input.mp4 output.mp4 --cell-size 15 --angle 30

  # Inverted: bright areas become big dots (white-on-black feel):
  python halftone_video.py input.mp4 output.mp4 --invert

  # Render only the first 60 frames for a quick preview:
  python halftone_video.py input.mp4 preview.mp4 --frames 60
""",
    )
    parser.add_argument("input",  help="Input video file")
    parser.add_argument("output", help="Output video file (.mp4 recommended)")
    parser.add_argument(
        "--mode", choices=["bw", "color"], default="bw",
        help="bw = grayscale dots on white  |  color = CMYK screen simulation  (default: bw)",
    )
    parser.add_argument(
        "--cell-size", type=int, default=10, metavar="PX",
        help="Halftone cell size in pixels — controls dot density (default: 10)",
    )
    parser.add_argument(
        "--angle", type=float, default=45.0, metavar="DEG",
        help="Grid angle in degrees — used in bw mode (default: 45)",
    )
    parser.add_argument(
        "--invert", action="store_true",
        help="Invert dot sizes: bright areas → large dots, dark areas → small dots",
    )
    parser.add_argument(
        "--frames", type=int, default=None, metavar="N",
        help="Process only the first N frames (useful for quick previews)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_video(
        input_path,
        output_path,
        mode=args.mode,
        cell_size=args.cell_size,
        angle=args.angle,
        invert=args.invert,
        max_frames=args.frames,
    )


if __name__ == "__main__":
    main()
