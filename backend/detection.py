import cv2
import json
import numpy as np
import math
import logging
import os
import time

log = logging.getLogger("bird_counter")

# ── Profiling flag ─────────────────────────────────────────────────────────────
_PROFILE = False  # Set True to enable per-stage timing instrumentation

# ── Processing efficiency ──────────────────────────────────────────────────────
FRAME_SKIP                       = 2     # process every 2nd frame — birds visible 4+ frames still tracked
RESIZE_FACTOR                    = 0.75  # 75% scale — cuts work ~44% while keeping distant birds as 1-2px blobs

# ── Appearance filter ──────────────────────────────────────────────────────────
MIN_BIRD_AREA                    = 0      # 1px dots (very distant) pass
MAX_BIRD_AREA                    = 80000  # large nearby birds (e.g. 300×250px) pass
MIN_ASPECT_RATIO                 = 0.08   # elongated distant shapes allowed
MAX_ASPECT_RATIO                 = 12.0
MIN_BIRD_SOLIDITY                = 0.1    # tiny blobs have unreliable solidity

# Adaptive threshold — two-pass: small block for distant dots, larger for close birds
ADAPTIVE_BLOCK_SIZE              = 5      # fine-grained; catches 1-3px blobs
ADAPTIVE_C                       = 2
ADAPTIVE_BLOCK_SIZE_LARGE        = 21     # coarser; merges fragmented large-bird contours
ADAPTIVE_C_LARGE                 = 4

# ── Bird colour filter ────────────────────────────────────────────────────────
# Birds are dark (grey/black) silhouettes. The contrast check pad scales with
# blob size so large birds aren't self-cancelling.
BIRD_MAX_BLOB_VALUE              = 170   # grey-to-black birds (HSV V 0-170); bright sky is 180-240
BIRD_MIN_CONTRAST                = 20    # surround must be clearly brighter than the bird
# Minimum blob side-length (px) above which we use the large-block threshold pass
LARGE_BIRD_MIN_SIDE              = 20

# ── Cloud rejection ────────────────────────────────────────────────────────────
CLOUD_TEXTURE_THRESHOLD          = 18    # avg std-dev of local intensity
CLOUD_EDGE_DENSITY_THRESHOLD     = 0.04  # fraction of edge pixels
CLOUD_COLOR_UNIFORMITY_THRESHOLD = 22    # std-dev of HSV saturation
TEXTURE_NEIGHBORHOOD             = 7

# ── Motion (optical flow) ──────────────────────────────────────────────────────
MOTION_THRESHOLD                 = 2     # very low — catches 1px shift between frames
MIN_MOTION_PIXELS                = 1
# Farneback params tuned for tiny/distant targets
_FB_PYR_SCALE                    = 0.5
_FB_LEVELS                       = 3     # 3 levels sufficient for local patch flow; 5 was overkill
_FB_WIN_SIZE                     = 7     # small window → fine-grained local motion
_FB_ITERATIONS                   = 3
_FB_POLY_N                       = 5
_FB_POLY_SIGMA                   = 1.1
MIN_BIRD_SPEED                   = 0.1   # near-zero: distant birds barely move in px/frame
MAX_BIRD_SPEED                   = 300
# Birds change direction; clouds drift uniformly
DIRECTIONAL_CHANGE_THRESHOLD     = 0.90  # relaxed — allow more uniform movers (distant birds)

# ── Background subtraction ─────────────────────────────────────────────────────
# MOG2 background subtractor catches any pixel-level movement in the sky.
_MOG2_HISTORY                    = 150   # slightly faster model update, no accuracy impact for birds
_MOG2_VAR_THRESHOLD              = 20    # not too sensitive — avoid sensor noise
_MOG2_DETECT_SHADOWS             = False # shadow pixels would inflate counts

# ── ROI — fixed: always use top 75% of frame ──────────────────────────────────
# Dynamic horizon detection was misidentifying dense bird flocks as the horizon
# and setting the ROI above them. Fixed ROI is simpler and more reliable.
ROI_UPDATE_INTERVAL              = 50    # kept for compatibility but ROI no longer changes
ROI_SMOOTHING_FACTOR             = 0.5
INITIAL_ROI_BOTTOM_PERCENTAGE    = 75    # fixed at 75% from top
ROI_MIN_PCT                      = 75.0  # min == max == fixed value
ROI_MAX_PCT                      = 75.0

# ── Temporal confirmation (tracker) ───────────────────────────────────────────
MIN_MOVEMENT_DISTANCE            = 6     # restored — 4px total across track lifetime
MIN_FLIGHT_DURATION              = 4     # restored original
MAX_MATCHING_DISTANCE            = 60    # px; large nearby birds move fast between frames
MAX_MATCHING_DISTANCE_CONFIRMED  = 120   # wider re-match for confirmed birds that briefly disappear
MAX_STATIONARY_FRAMES            = 8     # must survive enough gaps for intermittent far-bird detections
MAX_STATIONARY_FRAMES_CONFIRMED  = 20


# ── Helpers ────────────────────────────────────────────────────────────────────

def calculate_object_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area if hull_area > 0 else 0.0


def analyze_and_set_roi(frame, current_roi_bottom_pct, frame_gray=None):
    """
    Estimate the sky-ground horizon using Sobel edge projection and return a
    smoothed ROI bottom percentage.  This is the adaptive ROI strategy from
    the paper: Sobel gradient → vertical projection → search upward until
    edge density drops, then temporally smooth.

    frame_gray: optional precomputed grayscale of the full frame (avoids re-conversion).
    """
    height, width = frame.shape[:2]
    if frame_gray is not None:
        gray = frame_gray
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    vertical_projection = np.sum(sobel_y, axis=1)

    try:
        # Only search for the sky-ground boundary in the LOWER half of the frame.
        # Searching the full frame causes a dense flock of birds (strong Sobel response
        # in the upper sky) to be mistaken for the horizon, setting the ROI *above* the
        # flock and cutting it out entirely. Buildings / ground are always in the lower
        # portion of the frame.
        search_start = height // 2
        lower_projection = vertical_projection[search_start:]
        if len(lower_projection) == 0:
            return current_roi_bottom_pct

        peak_offset = int(np.argmax(lower_projection))
        peak_density_y = search_start + peak_offset

        if peak_density_y >= len(vertical_projection) - 1:
            return current_roi_bottom_pct

        # Search upward from the peak (the building/ground edge) to find where
        # edge density drops — that is the actual sky-ground transition.
        threshold = np.percentile(lower_projection, 10)
        horizon_y = peak_density_y
        for y in range(peak_density_y, search_start - 1, -1):
            if vertical_projection[y] < threshold:
                horizon_y = y
                break

        new_roi_bottom_pct = (horizon_y / height) * 100
        # Add 8% buffer so birds flying just above the roofline are inside the ROI.
        new_roi_bottom_pct = min(ROI_MAX_PCT, new_roi_bottom_pct + 8.0)
        smoothed = (current_roi_bottom_pct * (1.0 - ROI_SMOOTHING_FACTOR) +
                    new_roi_bottom_pct * ROI_SMOOTHING_FACTOR)
        return max(ROI_MIN_PCT, min(ROI_MAX_PCT, smoothed))
    except Exception:
        return current_roi_bottom_pct


def is_sky_color(frame, bbox, roi_bottom_pct, hsv_frame=None):
    """
    Multi-condition HSV sky classifier matching the four atmospheric states:
    clear blue, white/light-grey, light-blue/cyan, and grey/overcast.
    Checks the immediate region around the blob.
    """
    x, y, w, h = bbox
    frame_height, frame_width = frame.shape[:2]
    roi_bottom = int(frame_height * roi_bottom_pct / 100)
    if y > roi_bottom:
        return False

    border = 8
    x1 = max(0, x - border)
    y1 = max(0, y - border)
    x2 = min(frame_width, x + w + border)
    y2 = min(frame_height, y + h + border)

    if x2 <= x1 or y2 <= y1:
        return False

    if hsv_frame is not None:
        hsv_h, hsv_w = hsv_frame.shape[:2]
        hx1 = max(0, min(x1, hsv_w - 1))
        hy1 = max(0, min(y1, hsv_h - 1))
        hx2 = max(0, min(x2, hsv_w))
        hy2 = max(0, min(y2, hsv_h))
        if hx2 <= hx1 or hy2 <= hy1:
            return False
        hsv = hsv_frame[hy1:hy2, hx1:hx2]
    else:
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    if hsv.size == 0:
        return False

    avg_sat = float(np.mean(hsv[:, :, 1]))
    avg_val = float(np.mean(hsv[:, :, 2]))
    avg_hue = float(np.mean(hsv[:, :, 0]))

    # 1. White / light-grey sky (overcast or hazy)
    if avg_sat < 55 and avg_val > 150:
        return True
    # 2. Clear blue sky — widened sat upper bound for saturated afternoon blue
    # Fix 2: raised sat upper bound from 170 to 210 for deep saturated afternoon blue sky
    if 85 < avg_hue < 135 and avg_sat < 210 and avg_val > 75:
        return True
    # 3. Light blue / cyan sky
    if 75 < avg_hue < 110 and avg_sat < 180 and avg_val > 95:
        return True
    # 4. Grey / overcast
    if avg_sat < 28 and 70 < avg_val < 200:
        return True

    return False


def is_likely_cloud(frame, bbox, gray_roi=None, hsv_roi=None):
    """
    Multi-feature cloud rejection from the paper: simultaneously checks low
    texture variation, low edge density, and high colour uniformity — all
    characteristic of diffuse cloud surfaces rather than bird silhouettes.

    gray_roi: optional precomputed grayscale of roi_bgr (same spatial extent as frame).
    hsv_roi:  optional precomputed HSV of roi_bgr (same spatial extent as frame).
    When provided, patches are sliced from the precomputed arrays instead of re-converting.
    """
    x, y, w, h = bbox
    # Only run for regions large enough to be clouds; tiny blobs are never clouds
    if w < 8 or h < 8:
        return False
    fh, fw = frame.shape[:2]
    x  = max(0, x)
    y  = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x2 <= x or y2 <= y:
        return False

    # Grayscale patch
    if gray_roi is not None:
        gh, gw = gray_roi.shape[:2]
        gx1 = max(0, min(x, gw - 1))
        gy1 = max(0, min(y, gh - 1))
        gx2 = max(0, min(x2, gw))
        gy2 = max(0, min(y2, gh))
        if gx2 <= gx1 or gy2 <= gy1:
            return False
        gray_r = gray_roi[gy1:gy2, gx1:gx2]
    else:
        region = frame[y:y2, x:x2]
        if region.size == 0:
            return False
        gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    if gray_r.size == 0:
        return False

    # Texture: average local standard deviation
    k = np.ones((TEXTURE_NEIGHBORHOOD, TEXTURE_NEIGHBORHOOD), np.float32) / (TEXTURE_NEIGHBORHOOD ** 2)
    gf = gray_r.astype(np.float32)
    local_mean     = cv2.filter2D(gf, -1, k)
    local_sqr_mean = cv2.filter2D(gf ** 2, -1, k)
    variance       = np.clip(local_sqr_mean - local_mean ** 2, 0, None)
    avg_texture    = float(np.mean(np.sqrt(variance)))

    # Colour uniformity: std-dev of HSV saturation
    if hsv_roi is not None:
        hh, hw = hsv_roi.shape[:2]
        hx1 = max(0, min(x, hw - 1))
        hy1 = max(0, min(y, hh - 1))
        hx2 = max(0, min(x2, hw))
        hy2 = max(0, min(y2, hh))
        if hx2 <= hx1 or hy2 <= hy1:
            color_std = 0.0
        else:
            hsv_patch = hsv_roi[hy1:hy2, hx1:hx2]
            color_std = float(np.std(hsv_patch[:, :, 1])) if hsv_patch.size > 0 else 0.0
    else:
        region = frame[y:y2, x:x2]
        if region.size == 0:
            return False
        hsv_r    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        color_std = float(np.std(hsv_r[:, :, 1]))

    # Edge density
    edges        = cv2.Canny(gray_r, 50, 150)
    edge_density = np.count_nonzero(edges) / max(1, gray_r.size)

    return (avg_texture < CLOUD_TEXTURE_THRESHOLD
            and color_std < CLOUD_COLOR_UNIFORMITY_THRESHOLD
            and edge_density < CLOUD_EDGE_DENSITY_THRESHOLD)


def is_dark_against_sky(roi_bgr, bbox, gray_roi=None, hsv_roi=None):
    """
    Return True only if the blob is a dark (grey/black) object against a brighter sky.

    Fast path: use precomputed gray_roi for the cheap brightness gate, then slice
    directly from precomputed hsv_roi for the contrast check — avoids any BGR→HSV
    conversion per contour. The surround is sampled as four thin border strips around
    the blob rather than a masked ring, which is ~10× faster than numpy boolean masking.
    """
    x, y, w, h = bbox
    fh, fw = roi_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + max(1, w))
    y2 = min(fh, y + max(1, h))
    if x2 <= x1 or y2 <= y1:
        return False

    # --- Cheap gray pre-check ---
    if gray_roi is not None:
        gh, gw = gray_roi.shape[:2]
        gx1 = max(0, min(x1, gw - 1))
        gy1 = max(0, min(y1, gh - 1))
        gx2 = max(gx1 + 1, min(x2, gw))
        gy2 = max(gy1 + 1, min(y2, gh))
        blob_gray = gray_roi[gy1:gy2, gx1:gx2]
        if blob_gray.size > 0 and float(np.mean(blob_gray)) > BIRD_MAX_BLOB_VALUE:
            return False

    # --- Blob HSV value ---
    if hsv_roi is not None:
        hh, hw = hsv_roi.shape[:2]
        hx1 = max(0, min(x1, hw - 1))
        hy1 = max(0, min(y1, hh - 1))
        hx2 = max(hx1 + 1, min(x2, hw))
        hy2 = max(hy1 + 1, min(y2, hh))
        blob_v = hsv_roi[hy1:hy2, hx1:hx2, 2]
        blob_val = float(np.mean(blob_v)) if blob_v.size > 0 else 255.0
    else:
        blob_region = roi_bgr[y1:y2, x1:x2]
        if blob_region.size == 0:
            return False
        blob_val = float(np.mean(cv2.cvtColor(blob_region, cv2.COLOR_BGR2HSV)[:, :, 2]))

    if blob_val > BIRD_MAX_BLOB_VALUE:
        return False

    # --- Surround brightness via four border strips (no masking needed) ---
    pad = max(6, int(max(w, h) * 0.20))
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(fw, x2 + pad)
    sy2 = min(fh, y2 + pad)

    if hsv_roi is not None:
        hh, hw = hsv_roi.shape[:2]
        # Clamp surround to hsv_roi bounds
        hsx1 = max(0, min(sx1, hw - 1))
        hsy1 = max(0, min(sy1, hh - 1))
        hsx2 = max(hsx1 + 1, min(sx2, hw))
        hsy2 = max(hsy1 + 1, min(sy2, hh))
        hx1c = max(0, min(x1, hw - 1))
        hy1c = max(0, min(y1, hh - 1))
        hx2c = max(hx1c + 1, min(x2, hw))
        hy2c = max(hy1c + 1, min(y2, hh))
        # Four strips: top, bottom, left, right (may overlap at corners but that's fine)
        strips = []
        if hsy1 < hy1c:
            strips.append(hsv_roi[hsy1:hy1c, hsx1:hsx2, 2].ravel())
        if hy2c < hsy2:
            strips.append(hsv_roi[hy2c:hsy2, hsx1:hsx2, 2].ravel())
        if hsx1 < hx1c:
            strips.append(hsv_roi[hsy1:hsy2, hsx1:hx1c, 2].ravel())
        if hx2c < hsx2:
            strips.append(hsv_roi[hsy1:hsy2, hx2c:hsx2, 2].ravel())
        if not strips:
            return blob_val < BIRD_MAX_BLOB_VALUE
        outer_vals = np.concatenate(strips)
    else:
        full_patch = roi_bgr[sy1:sy2, sx1:sx2]
        if full_patch.size == 0:
            return blob_val < BIRD_MAX_BLOB_VALUE
        full_hsv = cv2.cvtColor(full_patch, cv2.COLOR_BGR2HSV)
        iy1 = y1 - sy1; ix1 = x1 - sx1
        iy2 = iy1 + (y2 - y1); ix2 = ix1 + (x2 - x1)
        mask = np.ones(full_patch.shape[:2], dtype=bool)
        mask[max(0,iy1):iy2, max(0,ix1):ix2] = False
        outer_vals = full_hsv[:, :, 2][mask]

    if outer_vals.size == 0:
        return blob_val < BIRD_MAX_BLOB_VALUE

    surround_val = float(np.percentile(outer_vals, 85))
    return (surround_val - blob_val) >= BIRD_MIN_CONTRAST


# ── MOG2 foreground mask → detections ─────────────────────────────────────────

def _extract_mog_detections(fg_mask, roi_bgr, gray_roi=None, hsv_roi=None):
    """
    Convert a MOG2 foreground mask into bird-candidate detections.
    Any pixel that changed against the learned background is a candidate.
    Tiny 1px blobs are allowed; only obvious clouds (large, low-texture) are rejected.

    gray_roi: optional precomputed grayscale (same spatial extent as roi_bgr).
    hsv_roi:  optional precomputed HSV (same spatial extent as roi_bgr).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (area >= MIN_BIRD_AREA and area < MAX_BIRD_AREA):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Large merged MOG2 blobs (flock seen as one connected region) are silently
        # dropped here — the adaptive threshold pipeline handles individual birds
        # within the region instead. Do NOT add them as a single detection.
        if w > 120 or h > 120:
            continue
        aspect_ratio = (w / h) if h > 0 else 1.0
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue
        if is_likely_cloud(roi_bgr, (x, y, w, h), gray_roi=gray_roi, hsv_roi=hsv_roi):
            continue
        if not is_dark_against_sky(roi_bgr, (x, y, w, h), gray_roi=gray_roi, hsv_roi=hsv_roi):
            continue
        detections.append({
            'bbox':         (x, y, max(1, w), max(1, h)),
            'center':       (x + max(1, w) // 2, y + max(1, h) // 2),
            'area':         float(area),
            'aspect_ratio': aspect_ratio,
            'confidence':   min(1.0, float(area) / 80.0),
        })
    return detections


# ── Appearance-based detection ─────────────────────────────────────────────────

def detect_birds_in_frame(frame, frame_height=None, detection_boundary=None, threshold=None):
    """
    Adaptive-threshold pipeline (best strategy per paper).
    1. Crop to ROI (sky region).
    2. Gaussian blur → adaptive Gaussian threshold (or fixed if supplied).
    3. Contour extraction with size / aspect-ratio / solidity / cloud filters.

    A second pass with coarse block and two upscale passes (1.5×, 2.0×) are run
    to catch birds that are too small to survive contour extraction at normal scale.

    Optimization: gray_roi is computed once before the passes and reused.
    For scaled passes, the precomputed gray is resized rather than re-converting BGR.
    """
    if frame_height is None:
        frame_height = frame.shape[0]
    if detection_boundary is None:
        detection_boundary = int(frame_height * INITIAL_ROI_BOTTOM_PERCENTAGE / 100)

    roi = frame[:detection_boundary, :]

    # Precompute gray and HSV once for the entire ROI — reused across all passes
    gray_roi_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi_base  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def _detect_at_scale(roi_bgr, scale=1.0, block_size=ADAPTIVE_BLOCK_SIZE,
                         c_val=ADAPTIVE_C, precomputed_gray=None):
        """
        precomputed_gray: if provided at scale==1.0, skip cvtColor.
                          if provided at scale!=1.0, resize this gray instead of
                          resizing BGR then converting.
        """
        if scale != 1.0:
            h0, w0 = roi_bgr.shape[:2]
            scaled_w = int(w0 * scale)
            scaled_h = int(h0 * scale)
            if precomputed_gray is not None:
                # Resize precomputed gray — avoids resizing BGR then converting
                gray = cv2.resize(precomputed_gray, (scaled_w, scaled_h),
                                  interpolation=cv2.INTER_LINEAR)
                # We still need a scaled BGR for is_dark_against_sky at native coords
                # but that check uses the original roi (unscaled), so no scaled BGR needed here
                roi_bgr_scaled = cv2.resize(roi_bgr, (scaled_w, scaled_h),
                                            interpolation=cv2.INTER_LINEAR)
            else:
                roi_bgr_scaled = cv2.resize(roi_bgr, (scaled_w, scaled_h),
                                            interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(roi_bgr_scaled, cv2.COLOR_BGR2GRAY)
        else:
            roi_bgr_scaled = roi_bgr
            if precomputed_gray is not None:
                gray = precomputed_gray
            else:
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Blur kernel scales with block size so large-block pass smooths coarser structure
        ksize   = max(3, (block_size // 3) | 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        if threshold is not None:
            _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size, c_val,
            )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (area >= MIN_BIRD_AREA and area < MAX_BIRD_AREA):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = (w / h) if h > 0 else 0
            if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO):
                continue

            # --- Cheap blob brightness pre-filter (uses gray slice, no HSV) ---
            inv = 1.0 / scale
            ox, oy = int(x * inv), int(y * inv)
            ow, oh = max(1, int(w * inv)), max(1, int(h * inv))

            # Fast gray pre-check on the native-resolution gray before expensive ops
            fh_g, fw_g = gray_roi_base.shape[:2]
            gx1 = max(0, ox)
            gy1 = max(0, oy)
            gx2 = min(fw_g, ox + ow)
            gy2 = min(fh_g, oy + oh)
            if gx2 > gx1 and gy2 > gy1:
                blob_gray_patch = gray_roi_base[gy1:gy2, gx1:gx2]
                if blob_gray_patch.size > 0:
                    if float(np.mean(blob_gray_patch)) > BIRD_MAX_BLOB_VALUE:
                        continue

            # --- Solidity check — skip convexHull for tiny single-pixel blobs ---
            if area < 4:
                solidity = 1.0
            else:
                solidity = calculate_object_solidity(contour)
            if solidity < MIN_BIRD_SOLIDITY:
                continue

            # --- Cloud rejection (expensive) ---
            if is_likely_cloud(roi_bgr_scaled, (x, y, w, h)):
                continue

            # --- Full contrast check at native resolution ---
            if not is_dark_against_sky(roi, (ox, oy, ow, oh),
                                       gray_roi=gray_roi_base, hsv_roi=hsv_roi_base):
                continue

            detections.append({
                'bbox':         (ox, oy, ow, oh),
                'center':       (ox + ow // 2, oy + oh // 2),
                'area':         area * (inv ** 2),
                'aspect_ratio': aspect_ratio,
                'confidence':   min(1.0, area / 80.0),
            })
        return detections

    def _merge(base, extras, tol=8):
        centres = [(d['center'][0], d['center'][1]) for d in base]
        for d in extras:
            cx, cy = d['center']
            if not any(abs(cx - ex) < tol and abs(cy - ey) < tol for ex, ey in centres):
                base.append(d)
                centres.append((cx, cy))

    # Pass A — fine block at native scale: catches 1-3px distant birds
    detections_all = _detect_at_scale(roi, scale=1.0,
                                      block_size=ADAPTIVE_BLOCK_SIZE,
                                      c_val=ADAPTIVE_C,
                                      precomputed_gray=gray_roi_base)

    # Pass B — 1.5× upscale: makes sub-pixel distant birds detectable as contours.
    # The 2× pass and coarse-block pass were dropped — they produced mostly duplicates
    # of Pass A / Pass B while doubling the adaptive threshold cost.
    detections_up = _detect_at_scale(roi, scale=1.5,
                                      block_size=ADAPTIVE_BLOCK_SIZE,
                                      c_val=ADAPTIVE_C,
                                      precomputed_gray=gray_roi_base)
    _merge(detections_all, detections_up, tol=6)

    return detections_all, gray_roi_base


# ── Motion-based discrimination ────────────────────────────────────────────────

def detect_moving_birds(current_detections, current_gray_roi, prev_gray_roi, roi_bgr):
    """
    Lucas-Kanade sparse optical flow — one batched PyrLK call across all candidates
    instead of N separate Farneback calls. ~10-20x faster on the flow stage.

    Falls back to frame differencing for any point LK fails to track.
    Cloud rejection still runs via is_likely_cloud for blobs with no clear motion vector.
    """
    if prev_gray_roi is None:
        return [], current_gray_roi

    if prev_gray_roi.shape != current_gray_roi.shape:
        return [], current_gray_roi

    if not current_detections:
        return [], current_gray_roi

    fh, fw = current_gray_roi.shape[:2]

    # Clamp all bbox centers to ROI bounds
    pts = []
    valid_detections = []
    for detection in current_detections:
        x, y, w, h = detection['bbox']
        cx = max(0, min(fw - 1, x + w // 2))
        cy = max(0, min(fh - 1, y + h // 2))
        pts.append([[float(cx), float(cy)]])
        valid_detections.append(detection)

    if not pts:
        return [], current_gray_roi

    pts_arr = np.array(pts, dtype=np.float32)

    pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray_roi, current_gray_roi, pts_arr, None,
        winSize=(9, 9),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    moving_birds = []
    for i, detection in enumerate(valid_detections):
        x, y, w, h = detection['bbox']
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        blob_area = (x2 - x1) * (y2 - y1)

        if status is not None and status[i, 0] == 1:
            # LK tracked successfully — compute displacement magnitude
            dx = float(pts_new[i, 0, 0] - pts_arr[i, 0, 0])
            dy = float(pts_new[i, 0, 1] - pts_arr[i, 0, 1])
            avg_magnitude = math.sqrt(dx * dx + dy * dy)

            # Frame diff for motion_pixels count (cheap, already needed)
            diff_region = cv2.absdiff(
                prev_gray_roi[y1:y2, x1:x2],
                current_gray_roi[y1:y2, x1:x2],
            )
            motion_pixels = int(np.count_nonzero(diff_region > MOTION_THRESHOLD))

            flow_magnitude_ok = MIN_BIRD_SPEED < avg_magnitude < MAX_BIRD_SPEED
            # For tiny far blobs the LK window is mostly sky — trust frame diff instead
            diff_only_ok = blob_area <= 6 and motion_pixels >= 1

            passes_motion = (
                motion_pixels >= MIN_MOTION_PIXELS
                and (flow_magnitude_ok or diff_only_ok)
                and not is_likely_cloud(roi_bgr, (x, y, w, h))
            )

            if passes_motion:
                detection['motion_score']    = avg_magnitude
                detection['motion_pixels']   = motion_pixels
                detection['flow_consistency'] = 0.5  # neutral: LK gives magnitude only
                moving_birds.append(detection)

        else:
            # LK failed (e.g. point near edge) — fall back to frame differencing
            diff_region = cv2.absdiff(
                prev_gray_roi[y1:y2, x1:x2],
                current_gray_roi[y1:y2, x1:x2],
            )
            motion_pixels = int(np.count_nonzero(diff_region > MOTION_THRESHOLD))

            if (motion_pixels >= MIN_MOTION_PIXELS
                    and not is_likely_cloud(roi_bgr, (x, y, w, h))):
                detection['motion_score']    = float(motion_pixels)
                detection['motion_pixels']   = motion_pixels
                detection['flow_consistency'] = 0.5
                moving_birds.append(detection)

    return moving_birds, current_gray_roi


# ── Farneback implementation (kept for easy revert) ───────────────────────────
# To revert: rename detect_moving_birds above to detect_moving_birds_lk,
#            and rename detect_moving_birds_farneback below to detect_moving_birds.

def detect_moving_birds_farneback(current_detections, current_gray_roi, prev_gray_roi, roi_bgr):
    """Original Farneback per-patch implementation."""
    if prev_gray_roi is None:
        return [], current_gray_roi

    if prev_gray_roi.shape != current_gray_roi.shape:
        return [], current_gray_roi

    fh, fw = current_gray_roi.shape[:2]
    moving_birds = []

    for detection in current_detections:
        x, y, w, h = detection['bbox']
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        if x2 <= x or y2 <= y:
            continue

        pad = min(max(4, max(w, h)), 16)
        px1 = max(0, x - pad)
        py1 = max(0, y - pad)
        px2 = min(fw, x2 + pad)
        py2 = min(fh, y2 + pad)

        if (px2 - px1) < 5 or (py2 - py1) < 5:
            px2 = min(fw, px1 + max(5, px2 - px1))
            py2 = min(fh, py1 + max(5, py2 - py1))
            if (px2 - px1) < 5 or (py2 - py1) < 5:
                continue

        prev_patch    = prev_gray_roi[py1:py2, px1:px2]
        current_patch = current_gray_roi[py1:py2, px1:px2]

        if prev_patch.size == 0 or current_patch.size == 0:
            continue

        patch_h, patch_w = prev_patch.shape[:2]
        win = max(3, min(_FB_WIN_SIZE, patch_w // 3, patch_h // 3) | 1)
        levels = max(1, min(_FB_LEVELS, int(math.log2(min(patch_w, patch_h) / 4 + 1))))

        patch_flow = cv2.calcOpticalFlowFarneback(
            prev_patch, current_patch, None,
            _FB_PYR_SCALE, levels, win,
            _FB_ITERATIONS, _FB_POLY_N, _FB_POLY_SIGMA, 0,
        )

        patch_diff = cv2.absdiff(prev_patch, current_patch)

        rel_x1 = max(0, x  - px1)
        rel_y1 = max(0, y  - py1)
        rel_x2 = min(px2 - px1, x2 - px1)
        rel_y2 = min(py2 - py1, y2 - py1)

        if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
            continue

        roi_diff = patch_diff[rel_y1:rel_y2, rel_x1:rel_x2]
        if roi_diff.size == 0:
            continue
        motion_pixels = int(np.count_nonzero(roi_diff > MOTION_THRESHOLD))

        roi_flow = patch_flow[rel_y1:rel_y2, rel_x1:rel_x2]
        if roi_flow.size == 0:
            continue
        mag, _ = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
        avg_magnitude = float(np.mean(mag))

        significant = np.count_nonzero(mag > 0.05)
        if significant > 4:
            flow_x = float(np.mean(roi_flow[..., 0]))
            flow_y = float(np.mean(roi_flow[..., 1]))
            flow_consistency = math.sqrt(flow_x ** 2 + flow_y ** 2) / (avg_magnitude + 1e-5)
        else:
            flow_consistency = 1.0

        blob_area = (x2 - x) * (y2 - y)
        flow_magnitude_ok = MIN_BIRD_SPEED < avg_magnitude < MAX_BIRD_SPEED
        diff_only_ok = blob_area <= 6 and motion_pixels >= 1

        passes_motion = (
            motion_pixels >= MIN_MOTION_PIXELS
            and (flow_magnitude_ok or diff_only_ok)
            and (
                flow_consistency < DIRECTIONAL_CHANGE_THRESHOLD
                or not is_likely_cloud(roi_bgr, (x, y, w, h))
            )
        )

        if passes_motion:
            detection['motion_score']    = avg_magnitude
            detection['motion_pixels']   = motion_pixels
            detection['flow_consistency'] = flow_consistency
            moving_birds.append(detection)

    return moving_birds, current_gray_roi


# ── Temporal confirmation tracker ─────────────────────────────────────────────

class ImprovedBirdTracker:
    def __init__(self):
        self.tracks                  = {}
        self.track_id                = 0
        self.confirmed_flying_birds  = set()   # active confirmed tracks only
        self._total_confirmed        = 0       # running count — never decrements

    def _is_duplicate_of_confirmed(self, x, y, exclude_tid=None):
        """
        Returns True only if (x,y) lands inside the bounding box of an existing
        confirmed track — meaning it is almost certainly a re-detection of the
        same individual, not a new bird.
        Using bbox overlap instead of a radius prevents suppressing neighboring
        birds in a dense flock that are close but spatially distinct.
        """
        for tid in self.confirmed_flying_birds:
            if tid == exclude_tid or tid not in self.tracks:
                continue
            bbox = self.tracks[tid].get('last_bbox')
            if bbox is None:
                continue
            bx, by, bw, bh = bbox
            # Expand bbox by 2px for tolerance
            if (bx - 2 <= x <= bx + bw + 2) and (by - 2 <= y <= by + bh + 2):
                return True
        return False

    def update_tracks(self, detections):
        matched_tracks     = {}
        current_frame_birds = []

        for detection in detections:
            x, y = detection['center']
            best_match, min_dist_sq = None, float('inf')

            for tid, tdata in self.tracks.items():
                if tid in matched_tracks or not tdata['positions']:
                    continue
                positions = tdata['positions']
                lx, ly    = positions[-1]

                # Velocity extrapolation for confirmed tracks that have been missing
                if (tid in self.confirmed_flying_birds
                        and tdata['stationary_count'] > 0
                        and len(positions) >= 2):
                    vx = positions[-1][0] - positions[-2][0]
                    vy = positions[-1][1] - positions[-2][1]
                    frames_missing = tdata['stationary_count']
                    lx = lx + vx * frames_missing
                    ly = ly + vy * frames_missing

                radius = (MAX_MATCHING_DISTANCE_CONFIRMED
                          if tid in self.confirmed_flying_birds
                          else MAX_MATCHING_DISTANCE)

                dx = x - lx
                dy = y - ly
                # Manhattan pre-filter: skip expensive sqrt if clearly out of range
                if abs(dx) > radius or abs(dy) > radius:
                    continue

                dist_sq = dx * dx + dy * dy
                radius_sq = radius * radius
                if dist_sq < min_dist_sq and dist_sq < radius_sq:
                    min_dist_sq, best_match = dist_sq, tid

            if best_match is not None:
                self.tracks[best_match]['positions'].append((x, y))
                self.tracks[best_match]['stationary_count'] = 0
                self.tracks[best_match]['detections'].append(detection)
                self.tracks[best_match]['last_bbox'] = detection['bbox']
                matched_tracks[best_match] = detection
                if (best_match not in self.confirmed_flying_birds
                        and self.is_flying_improved(best_match)):
                    if not self._is_duplicate_of_confirmed(x, y, exclude_tid=best_match):
                        self.confirmed_flying_birds.add(best_match)
                        self._total_confirmed += 1
                        current_frame_birds.append(detection)
                        # Bird is confirmed — history served its purpose, clear it now.
                        # Keep only last 2 positions for velocity extrapolation.
                        self.tracks[best_match]['positions'] = self.tracks[best_match]['positions'][-2:]
                        self.tracks[best_match]['detections'] = []
            else:
                # Always create a new track — _is_duplicate_of_confirmed only blocks
                # re-detections literally inside an existing confirmed bbox, not nearby birds
                if not self._is_duplicate_of_confirmed(x, y):
                    self.tracks[self.track_id] = {
                        'positions':       [(x, y)],
                        'stationary_count': 0,
                        'detections':      [detection],
                        'last_bbox':       detection['bbox'],
                    }
                    self.track_id += 1

        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid]['stationary_count'] += 1
                limit = (MAX_STATIONARY_FRAMES_CONFIRMED
                         if tid in self.confirmed_flying_birds
                         else MAX_STATIONARY_FRAMES)
                if self.tracks[tid]['stationary_count'] >= limit:
                    # Remove from confirmed set so _is_duplicate_of_confirmed never
                    # scans this dead track again — previously it stayed in the set forever.
                    self.confirmed_flying_birds.discard(tid)
                    del self.tracks[tid]

        return current_frame_birds

    def is_flying_improved(self, track_id):
        positions = self.tracks[track_id]['positions']
        if len(positions) < MIN_FLIGHT_DURATION:
            return False

        total_distance = 0.0
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_distance += math.sqrt(dx ** 2 + dy ** 2)
            if dx != 0 or dy != 0:
                directions.append(math.atan2(dy, dx))

        if total_distance < MIN_MOVEMENT_DISTANCE:
            return False

        direction_changes = 0
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i - 1])
            if diff > math.pi:
                diff = 2 * math.pi - diff
            if diff > 0.3:
                direction_changes += 1

        good_bird_features = 0
        dets   = self.tracks[track_id]['detections']
        scores = [d['motion_score'] for d in dets if 'motion_score' in d]
        if scores and np.mean(scores) > MIN_BIRD_SPEED:
            good_bird_features += 1
        flows = [d['flow_consistency'] for d in dets if 'flow_consistency' in d]
        if flows and np.mean(flows) < DIRECTIONAL_CHANGE_THRESHOLD:
            good_bird_features += 1
        ratios = [d['aspect_ratio'] for d in dets if 'aspect_ratio' in d]
        if ratios and np.std(ratios) > 0.1:
            good_bird_features += 1
        # Fix 4: A track seen across MIN_FLIGHT_DURATION+2 frames is itself evidence
        # of a real object. Far flock birds move in straight lines (no direction_changes)
        # and have near-zero flow scores — this gives them a confirmation path.
        if len(positions) >= MIN_FLIGHT_DURATION + 2:
            good_bird_features += 1

        return (direction_changes > 0 or good_bird_features >= 1)

    def get_unique_flying_birds_count(self):
        return self._total_confirmed

    def get_currently_active_birds(self):
        return len(self.confirmed_flying_birds & self.tracks.keys())

    def get_active_confirmed_bboxes(self):
        """Return one bbox per currently active confirmed track (no duplicates)."""
        result = []
        for tid in self.confirmed_flying_birds:
            if tid in self.tracks:
                bbox = self.tracks[tid].get('last_bbox')
                if bbox is not None:
                    result.append(bbox)
        return result


# ── Non-streaming entry point ──────────────────────────────────────────────────

def process_video_with_unique_bird_counting(video_path, show_display=False, threshold=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width    = int(width  * RESIZE_FACTOR)
    new_height   = int(height * RESIZE_FACTOR)

    print(f"Processing: {os.path.basename(video_path)}")
    print(f"- Resolution: {width}x{height} → {new_width}x{new_height}")
    print(f"- FPS: {fps}, Frames: {total_frames}, Skip: every {FRAME_SKIP}")

    bird_tracker      = ImprovedBirdTracker()
    bg_subtractor     = cv2.createBackgroundSubtractorMOG2(
        history=_MOG2_HISTORY,
        varThreshold=_MOG2_VAR_THRESHOLD,
        detectShadows=_MOG2_DETECT_SHADOWS,
    )
    roi_bottom_pct    = float(INITIAL_ROI_BOTTOM_PERCENTAGE)
    detection_boundary = int(new_height * roi_bottom_pct / 100.0)
    prev_frame_cache  = None
    frame_count       = 0
    max_concurrent_birds = 0
    start_time        = time.time()

    # Profile accumulators
    _p = {k: 0.0 for k in ('decode', 'roi_update', 'adaptive_threshold',
                             'mog2', 'sky_filter', 'optical_flow', 'tracking', 'display')}

    try:
        while True:
            for _ in range(FRAME_SKIP - 1):
                ret = cap.grab()
                if not ret:
                    break

            _t0 = time.perf_counter() if _PROFILE else 0
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += FRAME_SKIP

            if RESIZE_FACTOR != 1.0:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if _PROFILE:
                _p['decode'] += time.perf_counter() - _t0

            roi_bgr = frame[:detection_boundary, :]

            # Precompute ROI-level gray and HSV once — shared by sky_filter, mog2, etc.
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            roi_hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            _t0 = time.perf_counter() if _PROFILE else 0
            current_detections, current_gray_roi = detect_birds_in_frame(
                frame, new_height, detection_boundary, threshold=threshold
            )
            if _PROFILE:
                _p['adaptive_threshold'] += time.perf_counter() - _t0

            # MOG2 background subtraction: detect any pixel-level movement in the sky
            _t0 = time.perf_counter() if _PROFILE else 0
            fg_mask = bg_subtractor.apply(roi_bgr)
            mog_detections = _extract_mog_detections(fg_mask, roi_bgr,
                                                      gray_roi=roi_gray, hsv_roi=roi_hsv)
            existing = [(d['center'][0], d['center'][1]) for d in current_detections]
            for d in mog_detections:
                cx, cy = d['center']
                if not any(abs(cx - ex) < 6 and abs(cy - ey) < 6 for ex, ey in existing):
                    current_detections.append(d)
                    existing.append((cx, cy))
            if _PROFILE:
                _p['mog2'] += time.perf_counter() - _t0

            # Sky color filter — reuse precomputed roi_hsv (all candidates are within ROI)
            _t0 = time.perf_counter() if _PROFILE else 0
            sky_detections = [
                d for d in current_detections
                if is_sky_color(roi_bgr, d['bbox'], roi_bottom_pct, hsv_frame=roi_hsv)
            ]
            if _PROFILE:
                _p['sky_filter'] += time.perf_counter() - _t0

            _t0 = time.perf_counter() if _PROFILE else 0
            moving_detections, prev_frame_cache = detect_moving_birds(
                sky_detections, current_gray_roi, prev_frame_cache, roi_bgr
            )
            if _PROFILE:
                _p['optical_flow'] += time.perf_counter() - _t0

            _t0 = time.perf_counter() if _PROFILE else 0
            bird_tracker.update_tracks(moving_detections)
            if _PROFILE:
                _p['tracking'] += time.perf_counter() - _t0

            unique_flying_birds  = bird_tracker.get_unique_flying_birds_count()
            currently_active     = bird_tracker.get_currently_active_birds()
            max_concurrent_birds = max(max_concurrent_birds, currently_active)

            if frame_count % (100 * FRAME_SKIP) == 0:
                progress  = frame_count / total_frames if total_frames > 0 else 0
                elapsed   = time.time() - start_time
                remaining = (elapsed / progress - elapsed) if progress > 0 else 0
                print(
                    f"  {frame_count}/{total_frames} ({progress*100:.1f}%) — "
                    f"unique birds: {unique_flying_birds} — "
                    f"~{remaining:.0f}s remaining"
                )

        unique_flying_birds = bird_tracker.get_unique_flying_birds_count()
        total_time          = time.time() - start_time
        frames_per_second   = frame_count / total_time if total_time > 0 else 0

        if _PROFILE and frame_count > 0:
            total_stage = sum(_p.values())
            for stage, t in _p.items():
                ms_per_frame = t * 1000.0 / frame_count
                pct = (t / total_stage * 100) if total_stage > 0 else 0
                print(f"[profile] {stage}: {ms_per_frame:.2f}ms/frame ({pct:.1f}% total)")

        print(f"  DONE — {unique_flying_birds} unique flying birds in {total_time:.2f}s")

        return {
            'video_path':          video_path,
            'video_name':          os.path.basename(video_path),
            'frames_processed':    frame_count,
            'unique_flying_birds': unique_flying_birds,
            'max_concurrent_birds': max_concurrent_birds,
            'total_tracks':        bird_tracker.track_id,
            'fps':                 fps,
            'total_frames':        total_frames,
            'duration_seconds':    total_frames / fps if fps > 0 else 0,
            'processing_time':     total_time,
            'processing_speed':    frames_per_second,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None

    finally:
        cap.release()
        if show_display:
            cv2.destroyAllWindows()


# ── Streaming entry point ──────────────────────────────────────────────────────

_YELLOW              = (0, 255, 255)
_RED                 = (0, 0, 255)
_FRAME_SEND_INTERVAL = 15  # send annotated frame every N processed frames


def process_video_streaming(video_path, threshold=None):
    """
    Generator version of process_video_with_unique_bird_counting.
    Yields dicts:
      {"kind": "frame",    "frame": <annotated BGR ndarray>}
      {"kind": "progress", "progress": float, "unique_birds": int, "elapsed": float}
      {"kind": "result",   "data": {...}}
    """
    log.info("[detection] process_video_streaming START path=%s threshold=%s", video_path, threshold)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("[detection] cv2.VideoCapture failed to open: %s", video_path)
        return

    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width    = int(width  * RESIZE_FACTOR)
    new_height   = int(height * RESIZE_FACTOR)

    log.info(
        "[detection] video: %dx%d @ %dfps, %d frames, processing at %dx%d FRAME_SKIP=%d threshold=%s",
        width, height, fps, total_frames, new_width, new_height, FRAME_SKIP, threshold,
    )

    bird_tracker      = ImprovedBirdTracker()
    bg_subtractor     = cv2.createBackgroundSubtractorMOG2(
        history=_MOG2_HISTORY,
        varThreshold=_MOG2_VAR_THRESHOLD,
        detectShadows=_MOG2_DETECT_SHADOWS,
    )
    roi_bottom_pct    = float(INITIAL_ROI_BOTTOM_PERCENTAGE)
    detection_boundary = int(new_height * roi_bottom_pct / 100.0)
    prev_frame_cache  = None
    frame_count       = 0
    max_concurrent_birds = 0
    start_time        = time.time()

    first_detection_frame  = None
    peak_concurrent_frame  = 0
    birds_per_minute_counts: dict[int, int] = {}
    prev_unique            = 0
    total_motion_sum       = 0.0
    total_motion_count     = 0

    # Profile accumulators
    _p = {k: 0.0 for k in ('decode', 'roi_update', 'adaptive_threshold',
                             'mog2', 'sky_filter', 'optical_flow', 'tracking', 'display')}

    try:
        while True:
            for _ in range(FRAME_SKIP - 1):
                ret = cap.grab()
                if not ret:
                    break

            _t0 = time.perf_counter() if _PROFILE else 0
            ret, frame = cap.read()
            if not ret:
                if frame_count == 0:
                    log.error("[detection] cap.read() failed on first frame — video unreadable")
                else:
                    log.info("[detection] End of video at frame %d", frame_count)
                break

            frame_count += FRAME_SKIP
            if frame_count == FRAME_SKIP:
                log.info("[detection] First frame read OK shape=%s", frame.shape)

            if RESIZE_FACTOR != 1.0:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if _PROFILE:
                _p['decode'] += time.perf_counter() - _t0

            roi_bgr = frame[:detection_boundary, :]

            # Precompute ROI-level gray and HSV once per frame
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            roi_hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            _t0 = time.perf_counter() if _PROFILE else 0
            current_detections, current_gray_roi = detect_birds_in_frame(
                frame, new_height, detection_boundary, threshold=threshold
            )
            if _PROFILE:
                _p['adaptive_threshold'] += time.perf_counter() - _t0

            # MOG2 background subtraction: detect any pixel-level movement in the sky
            _t0 = time.perf_counter() if _PROFILE else 0
            fg_mask = bg_subtractor.apply(roi_bgr)
            mog_detections = _extract_mog_detections(fg_mask, roi_bgr,
                                                      gray_roi=roi_gray, hsv_roi=roi_hsv)
            existing = [(d['center'][0], d['center'][1]) for d in current_detections]
            for d in mog_detections:
                cx, cy = d['center']
                if not any(abs(cx - ex) < 6 and abs(cy - ey) < 6 for ex, ey in existing):
                    current_detections.append(d)
                    existing.append((cx, cy))
            if _PROFILE:
                _p['mog2'] += time.perf_counter() - _t0

            # Sky color filter — reuse precomputed roi_hsv (all candidates are within ROI)
            _t0 = time.perf_counter() if _PROFILE else 0
            sky_detections = [
                d for d in current_detections
                if is_sky_color(roi_bgr, d['bbox'], roi_bottom_pct, hsv_frame=roi_hsv)
            ]
            if _PROFILE:
                _p['sky_filter'] += time.perf_counter() - _t0

            _t0 = time.perf_counter() if _PROFILE else 0
            moving_detections, prev_frame_cache = detect_moving_birds(
                sky_detections, current_gray_roi, prev_frame_cache, roi_bgr
            )
            if _PROFILE:
                _p['optical_flow'] += time.perf_counter() - _t0

            _t0 = time.perf_counter() if _PROFILE else 0
            bird_tracker.update_tracks(moving_detections)
            if _PROFILE:
                _p['tracking'] += time.perf_counter() - _t0

            unique_flying_birds = bird_tracker.get_unique_flying_birds_count()
            currently_active    = bird_tracker.get_currently_active_birds()

            if first_detection_frame is None and unique_flying_birds > 0:
                first_detection_frame = frame_count

            if currently_active > max_concurrent_birds:
                max_concurrent_birds  = currently_active
                peak_concurrent_frame = frame_count

            if unique_flying_birds > prev_unique:
                if fps > 0:
                    minute_idx = int(frame_count / fps / 60)
                    new_birds  = unique_flying_birds - prev_unique
                    birds_per_minute_counts[minute_idx] = (
                        birds_per_minute_counts.get(minute_idx, 0) + new_birds
                    )
                prev_unique = unique_flying_birds

            for det in moving_detections:
                if 'motion_score' in det:
                    total_motion_sum   += det['motion_score']
                    total_motion_count += 1

            if frame_count == FRAME_SKIP:
                log.info(
                    "[detection] Frame 1 processed: detections=%d sky=%d moving=%d",
                    len(current_detections), len(sky_detections), len(moving_detections),
                )

            _t0 = time.perf_counter() if _PROFILE else 0
            if frame_count % _FRAME_SEND_INTERVAL == 0:
                display = frame.copy()

                # Draw confirmed active tracks — box follows the bird each frame
                confirmed_bboxes = bird_tracker.get_active_confirmed_bboxes()
                confirmed_centers = []
                for (x, y, w, h) in confirmed_bboxes:
                    cv2.rectangle(display, (x, y), (x + w, y + h), _RED, 2)
                    confirmed_centers.append((x + w // 2, y + h // 2))

                # Draw unconfirmed candidates only if not overlapping a confirmed track
                for det in moving_detections:
                    cx, cy = det['center']
                    near_confirmed = any(
                        abs(cx - ex) < MAX_MATCHING_DISTANCE and abs(cy - ey) < MAX_MATCHING_DISTANCE
                        for ex, ey in confirmed_centers
                    )
                    if not near_confirmed:
                        x, y, w, h = det['bbox']
                        cv2.rectangle(display, (x, y), (x + w, y + h), _YELLOW, 1)

                cv2.putText(
                    display,
                    f"Unique birds: {unique_flying_birds}  Active: {currently_active}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                yield {"kind": "frame", "frame": display}
            if _PROFILE:
                _p['display'] += time.perf_counter() - _t0

            if frame_count % (100 * FRAME_SKIP) == 0:
                elapsed = time.time() - start_time
                if total_frames > 0:
                    progress = frame_count / total_frames
                elif elapsed > 0:
                    progress = min(0.99, frame_count / max(1, frame_count + 500))
                else:
                    progress = 0
                yield {
                    "kind":         "progress",
                    "progress":     round(progress, 4),
                    "unique_birds": unique_flying_birds,
                    "elapsed":      round(elapsed, 1),
                }

        unique_flying_birds = bird_tracker.get_unique_flying_birds_count()
        total_time          = time.time() - start_time
        log.info(
            "[detection] DONE: %d unique birds, %d frames in %.1fs (%.1f fps processed)",
            unique_flying_birds, frame_count, total_time,
            frame_count / total_time if total_time > 0 else 0,
        )

        if _PROFILE and frame_count > 0:
            total_stage = sum(_p.values())
            for stage, t in _p.items():
                ms_per_frame = t * 1000.0 / frame_count
                pct = (t / total_stage * 100) if total_stage > 0 else 0
                log.info("[profile] %s: %.2fms/frame (%.1f%% total)", stage, ms_per_frame, pct)

        total_minutes = (int(frame_count / fps / 60) + 1) if fps > 0 else 0
        minute_counts = [birds_per_minute_counts.get(m, 0) for m in range(total_minutes)]

        yield {
            "kind": "result",
            "data": {
                "unique_flying_birds":    unique_flying_birds,
                "max_concurrent_birds":   max_concurrent_birds,
                "frames_processed":       frame_count,
                "total_tracks":           bird_tracker.track_id,
                "fps":                    fps,
                "total_frames":           total_frames,
                "duration_seconds":       (
                    (total_frames / fps) if (fps > 0 and total_frames > 0)
                    else round(total_time, 1)
                ),
                "processing_time":        round(total_time, 1),
                "processing_speed":       round(frame_count / total_time, 2) if total_time > 0 else 0,
                "first_detection_second": (
                    round(first_detection_frame / fps, 1)
                    if first_detection_frame and fps > 0 else None
                ),
                "peak_concurrent_second": round(peak_concurrent_frame / fps, 1) if fps > 0 else 0,
                "birds_per_minute":       json.dumps(minute_counts),
                "track_noise_ratio":      round(unique_flying_birds / max(1, bird_tracker.track_id), 3),
                "avg_motion_score":       round(total_motion_sum / max(1, total_motion_count), 2),
            },
        }

    except Exception as e:
        log.exception("[detection] Streaming error at frame %d: %s", frame_count, e)

    finally:
        cap.release()
        log.info("[detection] process_video_streaming END")
