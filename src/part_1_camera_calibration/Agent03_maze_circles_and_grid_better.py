#!/usr/bin/env python3
"""
maze_circles_and_grid.py

Workflow:
  1) Detect circles via HoughCircles and classify color (red/green/unknown).
     Saves a circles overlay.
  2) Build grid over the maze, detect black walls, and compute per-cell occupancy:
        value = 1 -> no wall in the cell
        value = 0 -> wall present in the cell (>= threshold % of wall pixels)
     Saves:
        - grid overlay (lines only)
        - annotated grid overlay (0/1 at cell centers)
        - walls mask (255 = wall)
  3) Write a single JSON with:
        - circles: [ {center:[x,y], radius:r, color:"red/green/unknown"} ]
        - grid_size_px, grid_rows, grid_cols, threshold_percent
        - paths to overlays and mask
        - cells: [ {row, col, value, center_px:[x,y]} ]

Usage:
  python maze_circles_and_grid.py maze.png \
    --grid 30 --threshold 0.0 \
    --circles-overlay-out circles_overlay.png \
    --grid-overlay-out grid_overlay.png \
    --grid-overlay-annot-out grid_overlay_annot.png \
    --walls-mask-out walls_mask.png \
    --json-out result.json \
    --adaptive 0 --blur 5 --open 0 --close 0 \
    --font-scale 0.4 --thickness 1
"""

import argparse
import sys
import json
import numpy as np
import cv2

# ---------------- Circle color detection ----------------

def detect_color(frame, center, radius):
    """Detect dominant color (red or green) inside a circle."""
    # Robust 80% inner-disk mask to avoid edge noise
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(max(1.0, radius * 0.8)), 255, -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Mask within circle
    H = H[mask == 255]
    S = S[mask == 255]
    V = V[mask == 255]

    # Filter only valid pixels (saturated + bright enough)
    valid = (S > 50) & (V > 50)
    H = H[valid]

    if len(H) == 0:
        return "green"

    # Count pixels in red and green hue ranges
    red_mask = ((H <= 10) | (H >= 170))
    green_mask = ((H >= 35) & (H <= 85))

    red_ratio = np.sum(red_mask) / len(H) if len(H) > 0 else 0.0
    green_ratio = np.sum(green_mask) / len(H) if len(H) > 0 else 0.0

    if red_ratio > green_ratio and red_ratio > 0.1:
        return "red"
    elif green_ratio > red_ratio and green_ratio > 0.1:
        return "green"
    else:
        return "unknown"

def detect_circles_and_overlay(img_bgr, out_path):
    """Detect circles, label with color, draw overlay; return list of dicts with center/radius/color."""
    frame = img_bgr.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0].astype(np.float32)
        circles = circles[np.argsort(circles[:, 0])]  # sort left to right

        for (u, v, r) in circles:
            color = detect_color(frame, (u, v), r)
            results.append({
                "center": [float(u), float(v)],
                "radius": float(r),
                "color": color
            })
            # draw the outer circle
            cv2.circle(frame, (int(u), int(v)), int(r), (0, 255, 0), 2)
            # draw the center
            cv2.circle(frame, (int(u), int(v)), 2, (0, 255, 0), 3)
            # label color
            cv2.putText(frame, color, (int(u) + 10, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # Save overlay
    cv2.imwrite(out_path, frame)
    return results

# ---------------- Grid + walls ----------------

def binarize_walls(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    """
    Return mask_walls (uint8) where 255 = wall (black in original), 0 = not wall.
    Otsu/Adaptive threshold with inversion.
    """
    if adaptive:
        block = 35 if min(gray.shape[:2]) > 400 else 21
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, block, 5)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def morph(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    m = mask.copy()
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_open+1, 2*k_open+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_close+1, 2*k_close+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m

def draw_grid_lines(img: np.ndarray, grid: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = (255, 255, 255)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w-1, y), color, 1)
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h-1), color, 1)
    return out

def draw_grid_with_values(img: np.ndarray, grid: int, values_mat: np.ndarray,
                          font_scale: float, thickness: int) -> np.ndarray:
    out = draw_grid_lines(img, grid)
    h, w = out.shape[:2]
    gh, gw = values_mat.shape
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            text = str(int(values_mat[gy, gx]))
            # black outline
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            # white text
            cv2.putText(out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return out

# ---- NEW: helpers for robust circle detection ----
def _preprocess_for_circles(img_bgr, clahe_clip=2.0, clahe_grid=8, blur=5):
    """Light normalization + contrast boost + gentle blur."""
    img = img_bgr.copy()

    # White balance (Gray World)
    avg_b = img[...,0].mean()
    avg_g = img[...,1].mean()
    avg_r = img[...,2].mean()
    gain_b = (avg_g + avg_r) / (2.0*avg_b + 1e-6)
    gain_g = (avg_b + avg_r) / (2.0*avg_g + 1e-6)
    gain_r = (avg_b + avg_g) / (2.0*avg_r + 1e-6)
    img = cv2.merge([
        np.clip(img[...,0]*gain_b, 0, 255).astype(np.uint8),
        np.clip(img[...,1]*gain_g, 0, 255).astype(np.uint8),
        np.clip(img[...,2]*gain_r, 0, 255).astype(np.uint8),
    ])

    # CLAHE on V channel to stabilize exposure
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    v = clahe.apply(v)
    img = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

    # Optional median blur to remove salt/pepper noise
    if blur > 0:
        k = max(1, blur | 1)
        img = cv2.medianBlur(img, k)

    return img

def _color_masks(img_bgr):
    """Return binary masks for red and green regions (robust to light)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broad but safe ranges
    # Red wraps around: lower [0-10], upper [170-180]
    red1 = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 40), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    green = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))

    # Clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    red  = cv2.morphologyEx(red,  cv2.MORPH_OPEN, k, iterations=1)
    green= cv2.morphologyEx(green,cv2.MORPH_OPEN, k, iterations=1)
    red  = cv2.morphologyEx(red,  cv2.MORPH_CLOSE, k, iterations=1)
    green= cv2.morphologyEx(green,cv2.MORPH_CLOSE, k, iterations=1)

    return red, green

def _hough_multi(gray, minR, maxR, minDist, dp, p1, p2):
    """Try classic HOUGH_GRADIENT and, if available, HOUGH_GRADIENT_ALT."""
    circles_all = []

    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
    if c is not None:
        circles_all.append(c[0])

    # ALT (OpenCV 4.5+) is often better on low-contrast/tilted blobs
    try:
        c2 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=dp, minDist=minDist,
                              param1=p1, param2=p2*0.6, minRadius=minR, maxRadius=maxR)
        if c2 is not None:
            circles_all.append(c2[0])
    except Exception:
        pass

    if not circles_all:
        return None

    circles = np.vstack(circles_all)
    return circles

def _contour_fallback(mask, minR, maxR):
    """When circles look elliptical, fall back to contour circularity."""
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < np.pi*(minR**2)*0.2:  # too tiny
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4*np.pi*area/(perim*perim + 1e-6)  # 1.0 is a perfect circle
        if circularity < 0.55:  # allow ellipses from perspective
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        if r < minR or r > maxR:
            continue
        found.append([x,y,r])
    if not found:
        return None
    return np.array(found, dtype=np.float32)

def _nms_circles(circles, iou_thr=0.5):
    """Non-max suppression on circles to merge duplicates from multiple methods."""
    if circles is None or len(circles)==0:
        return circles
    # format: [x,y,r] (float)
    c = circles.copy()
    scores = c[:,2]  # prefer larger radius (usually more stable)
    order = np.argsort(-scores)
    keep = []
    def iou(a,b):
        # approximate IoU using distance of centers vs radii (fast heuristic)
        da = np.hypot(a[0]-b[0], a[1]-b[1])
        if da >= a[2]+b[2]:
            return 0.0
        inter = min(a[2], b[2])**2  # cheap overlap proxy
        union = a[2]**2 + b[2]**2 - inter
        return inter/union
    for idx in order:
        good = True
        for j in keep:
            if iou(c[idx], c[j]) > iou_thr:
                good = False
                break
        if good:
            keep.append(idx)
    return c[keep]

# ---- UPDATED: more robust circle detection with color guidance + fallbacks ----
def detect_circles_and_overlay(img_bgr, out_path,
                               dp=1.2, minDist=40,
                               minRadius=None, maxRadius=None,
                               clahe_clip=2.0, clahe_grid=8, blur=5):
    """
    Detect circles under tilt and uneven lighting.
    Strategy:
      1) Preprocess (white balance + CLAHE + blur)
      2) Build red/green masks; run Hough on masked gray
      3) Also run Hough on global gray
      4) Fallback: contour-based detection on color masks
      5) Merge results (NMS) and color-classify
    """
    h, w = img_bgr.shape[:2]
    if minRadius is None or maxRadius is None:
        # Auto scale by image size (works for phone/robot cams)
        longest = max(h, w)
        minRadius = int(max(6, 0.01*longest)) if minRadius is None else minRadius
        maxRadius = int(max(12, 0.08*longest)) if maxRadius is None else maxRadius

    img_pp = _preprocess_for_circles(img_bgr, clahe_clip, clahe_grid, blur)
    red_mask, green_mask = _color_masks(img_pp)

    # Grayscale inputs
    gray_global = cv2.cvtColor(img_pp, cv2.COLOR_BGR2GRAY)

    # Focused grays: only where color says interesting
    gray_red   = cv2.bitwise_and(gray_global, gray_global, mask=red_mask)
    gray_green = cv2.bitwise_and(gray_global, gray_global, mask=green_mask)

    circles_list = []

    # Hough on global gray
    c_global = _hough_multi(gray_global, minRadius, maxRadius, minDist, dp, 120, 22)
    if c_global is not None:
        circles_list.append(c_global)

    # Hough on color-focused regions (lets us lower thresholds)
    c_red = _hough_multi(gray_red, minRadius, maxRadius, max(20, minDist//2), dp, 120, 18)
    if c_red is not None:
        circles_list.append(c_red)
    c_green = _hough_multi(gray_green, minRadius, maxRadius, max(20, minDist//2), dp, 120, 18)
    if c_green is not None:
        circles_list.append(c_green)

    # Fallback: contour-based
    c_cont_red   = _contour_fallback(red_mask,   minRadius, maxRadius)
    c_cont_green = _contour_fallback(green_mask, minRadius, maxRadius)
    for c in (c_cont_red, c_cont_green):
        if c is not None:
            circles_list.append(c)

    if circles_list:
        circles = np.vstack(circles_list).astype(np.float32)
        circles = _nms_circles(circles, iou_thr=0.5)
    else:
        circles = None

    # Draw + color label
    frame = img_bgr.copy()
    results = []
    if circles is not None and len(circles) > 0:
        # sort left-to-right for stable ordering
        circles = circles[np.argsort(circles[:,0])]
        for (u, v, r) in circles:
            color = detect_color(frame, (u, v), r)  # reuse your robust color vote
            results.append({
                "center": [float(u), float(v)],
                "radius": float(r),
                "color": color
            })
            cv2.circle(frame, (int(u), int(v)), int(r), (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 2, (0, 255, 0), 3)
            cv2.putText(frame, color, (int(u) + 10, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)
    return results

def Agent03_maze_circles_and_grid_better():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = "maze_warp.png", help="Path to maze image (white path, black wall)")
    ap.add_argument("--grid", type=int, default=30, help="Cell size in pixels")
    ap.add_argument("--adaptive", type=int, default=0, help="Use adaptive threshold (1) or Otsu (0, default)")
    ap.add_argument("--blur", type=int, default=5, help="Median blur kernel size (0=off)")
    ap.add_argument("--open", type=int, default=0, help="Morph open radius (px) to remove specks")
    ap.add_argument("--close", type=int, default=0, help="Morph close radius (px) to bridge gaps")
    ap.add_argument("--threshold", type=float, default=5.0, help="Percent of wall pixels to mark a cell as blocked (0..100)")

    ap.add_argument("--circles-overlay-out", default="circles_overlay.png", help="Detected circles + color labels")
    ap.add_argument("--grid-overlay-out", default="grid_overlay.png", help="Grid overlay (lines only)")
    ap.add_argument("--grid-overlay-annot-out", default="grid_overlay_annot.png", help="Grid overlay with 0/1 annotations")
    ap.add_argument("--walls-mask-out", default="walls_mask.png", help="Binary WALL mask (255=wall)")
    ap.add_argument("--json-out", default="result.json", help="Unified JSON with circles + per-cell data")

    ap.add_argument("--font-scale", type=float, default=0.4, help="Font scale for numbers")
    ap.add_argument("--thickness", type=int, default=1, help="Font thickness for numbers")

    ap.add_argument("--circle-dp", type=float, default=1.2, help="Hough dp (image pyramid inverse ratio)")
    ap.add_argument("--circle-min-dist", type=int, default=40, help="Minimum distance between circle centers")
    ap.add_argument("--circle-min-radius", type=int, default=0, help="Min circle radius (0=auto)")
    ap.add_argument("--circle-max-radius", type=int, default=0, help="Max circle radius (0=auto)")
    ap.add_argument("--circle-clahe-clip", type=float, default=2.0, help="CLAHE clip limit for circle preprocessing")
    ap.add_argument("--circle-clahe-grid", type=int, default=8, help="CLAHE grid size for circle preprocessing")


    args = ap.parse_args()

    # --- Load image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    grid = max(1, args.grid)

    # 1) Detect circles + colors; save overlay
    # circles_info = detect_circles_and_overlay(img, args.circles_overlay_out)
    circles_info = detect_circles_and_overlay(
        img,
        args.circles_overlay_out,
        dp=args.circle_dp,
        minDist=args.circle_min_dist,
        minRadius=(None if args.circle_min_radius<=0 else args.circle_min_radius),
        maxRadius=(None if args.circle_max_radius<=0 else args.circle_max_radius),
        clahe_clip=args.circle_clahe_clip,
        clahe_grid=args.circle_clahe_grid,
        blur=args.blur
    )


    # 2) GRID overlays + walls
    # 2a) Save grid overlay (lines only)
    grid_overlay = draw_grid_lines(img, grid)
    cv2.imwrite(args.grid_overlay_out, grid_overlay)

    # 2b) Walls mask (255=wall)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur > 0:
        k = max(1, args.blur | 1)  # ensure odd
        gray = cv2.medianBlur(gray, k)
    walls_mask = binarize_walls(gray, adaptive=bool(args.adaptive))
    walls_mask = morph(walls_mask, k_open=max(0, args.open), k_close=max(0, args.close))
    cv2.imwrite(args.walls_mask_out, walls_mask)

    # 2c) Per-cell values & centers
    h, w = walls_mask.shape[:2]
    gh = (h + grid - 1) // grid
    gw = (w + grid - 1) // grid
    values_mat = np.zeros((gh, gw), dtype=np.uint8)
    cells = []
    thresh_pct = float(args.threshold)

    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            blk = walls_mask[y0:y1, x0:x1]
            wall_pct = (blk > 0).mean() * 100.0 if blk.size > 0 else 0.0
            value = 0 if wall_pct >= thresh_pct else 1
            values_mat[gy, gx] = value
            cells.append({"row": int(gy), "col": int(gx), "value": int(value), "center_px": [int(cx), int(cy)]})

    # 2d) Annotated overlay (grid + 0/1 at centers)
    grid_overlay_annot = draw_grid_with_values(img, grid, values_mat, args.font_scale, args.thickness)
    cv2.imwrite(args.grid_overlay_annot_out, grid_overlay_annot)

    # 3) JSON export
    meta = {
        "input": args.input,
        "circles_overlay_path": args.circles_overlay_out,
        "grid_size_px": grid,
        "grid_rows": int(gh),
        "grid_cols": int(gw),
        "threshold_percent": thresh_pct,
        "grid_overlay_path": args.grid_overlay_out,
        "grid_overlay_annot_path": args.grid_overlay_annot_out,
        "walls_mask_path": args.walls_mask_out,
        "circles": circles_info,
        "cells": cells
    }
    with open(args.json_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Circles overlay saved to: {args.circles_overlay_out} ({len(circles_info)} circles)")
    print(f"Grid overlay saved to: {args.grid_overlay_out}")
    print(f"Walls mask saved to: {args.walls_mask_out}")
    print(f"Annotated grid overlay saved to: {args.grid_overlay_annot_out}")
    print(f"JSON saved to: {args.json_out}")
    print("Cell value legend: 1 = no wall, 0 = wall present")

# if __name__ == "__main__":
#     main()
