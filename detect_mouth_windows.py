import numpy as np
import srt

# --------------------------------
# Models
# --------------------------------
FACE_DETECTOR_MODEL = "models/mediapipe_models/detector.tflite"
FACE_LANDMARKER_MODEL = "models/mediapipe_models/face_landmarker.task"
MAX_DIM = 1280

# MediaPipe landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# --------------------------------
# Helpers
# --------------------------------
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def moving_average(arr, k=5):
    if len(arr) < k:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(k) / k, mode="same")


def build_windows(times, flags, min_len_ms=300):
    windows = []
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = times[i]
        elif not flag and start is not None:
            end = times[i]
            if end - start >= min_len_ms:
                windows.append({"start_ms": int(start), "end_ms": int(end)})
            start = None

    if start is not None:
        windows.append({"start_ms": int(start), "end_ms": int(times[-1])})

    return windows


def merge_windows(windows, max_gap_ms=200):
    if not windows:
        return []

    merged = [windows[0].copy()]

    for w in windows[1:]:
        prev = merged[-1]
        if w["start_ms"] - prev["end_ms"] <= max_gap_ms:
            prev["end_ms"] = w["end_ms"]
        else:
            merged.append(w.copy())

    return merged


def roi_motion_score(prev_roi, curr_roi):
    import cv2
    if prev_roi is None or curr_roi is None:
        return 0.0

    if prev_roi.shape != curr_roi.shape:
        curr_roi = cv2.resize(curr_roi, (prev_roi.shape[1], prev_roi.shape[0]))

    diff = cv2.absdiff(prev_roi, curr_roi)
    return float(np.mean(diff)) / 255.0


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def pick_biggest_detection(detections):
    if not detections:
        return None

    best = None
    best_area = -1

    for det in detections:
        bbox = det.bounding_box
        area = bbox.width * bbox.height
        if area > best_area:
            best_area = area
            best = det

    return best


def crop_face_with_padding(frame, bbox, pad_ratio=0.35):
    """
    bbox is MediaPipe FaceDetector bounding_box in pixels:
      origin_x, origin_y, width, height
    """
    h, w = frame.shape[:2]

    x = bbox.origin_x
    y = bbox.origin_y
    bw = bbox.width
    bh = bbox.height

    x1 = int(x - bw * pad_ratio)
    y1 = int(y - bh * pad_ratio)
    x2 = int(x + bw * (1.0 + pad_ratio))
    y2 = int(y + bh * (1.0 + pad_ratio))

    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 1, w)
    y2 = clamp(y2, 1, h)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


def crop_mouth_roi(frame, face_landmarks, frame_w, frame_h, pad=0.35):
    import cv2
    upper = face_landmarks[UPPER_LIP]
    lower = face_landmarks[LOWER_LIP]
    left = face_landmarks[LEFT_MOUTH]
    right = face_landmarks[RIGHT_MOUTH]

    pts = [upper, lower, left, right]
    xs = [p.x * frame_w for p in pts]
    ys = [p.y * frame_h for p in pts]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1

    if w <= 1 or h <= 1:
        return None

    x1 = max(0, int(x1 - w * pad))
    x2 = min(frame_w, int(x2 + w * pad))
    y1 = max(0, int(y1 - h * (pad + 0.2)))
    y2 = min(frame_h, int(y2 + h * (pad + 0.4)))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi


def build_search_views(frame):
    h, w = frame.shape[:2]
    views = []

    # 1) full frame
    views.append((frame, "full"))

    # 2) center crop ~70%
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    y1 = int(h * 0.10)
    y2 = int(h * 0.90)
    crop70 = frame[y1:y2, x1:x2]
    if crop70.size > 0:
        views.append((crop70, "center70"))

    # 3) center crop ~50%
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.18)
    y2 = int(h * 0.82)
    crop50 = frame[y1:y2, x1:x2]
    if crop50.size > 0:
        views.append((crop50, "center50"))

    return views


def detect_face_in_views(face_detector, search_views):
    """
    Returns:
      selected_view, selected_detection, selected_view_name, detections_count
    """
    import cv2
    import mediapipe as mp
    total_detections = 0

    for view_bgr, view_name in search_views:
        rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = face_detector.detect(mp_image)
        detections = result.detections if result.detections else []
        total_detections += len(detections)

        if detections:
            selected_detection = pick_biggest_detection(detections)
            return view_bgr, selected_detection, view_name, total_detections

    return None, None, None, total_detections


# --------------------------------
# Main
# --------------------------------
def detect_mouth_windows(video_path, subtitles_file,  sample_fps=4.0):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    # Face detector in IMAGE mode because we run it on multiple views of the same frame
    detector_base_options = python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL)
    detector_options = vision.FaceDetectorOptions(
        base_options=detector_base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.25,
    )
    face_detector = vision.FaceDetector.create_from_options(detector_options)

    # Face landmarker on cropped face
    landmarker_base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL)
    landmarker_options = vision.FaceLandmarkerOptions(
        base_options=landmarker_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        face_detector.close()
        face_landmarker.close()
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        sample_every_n_frames = max(1, int(round(fps / sample_fps)))

        frame_idx = 0

        mouth_open = []
        motion = []
        roi_motion = []
        visible = []
        times = []

        prev_open = None
        prev_mouth_roi = None

        while True:
            ok = cap.grab()
            if not ok:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)

            # only process sampled frames
            if frame_idx % sample_every_n_frames != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break

            # Optional resize for large videos
            h, w = frame.shape[:2]
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # 1) build search views
            search_views = build_search_views(frame)

            # 2) detect face in views
            selected_view, selected_detection, selected_view_name, detections_count = detect_face_in_views(
                face_detector, search_views
            )

            if selected_detection is None or selected_view is None:
                visible.append(False)
                mouth_open.append(0.0)
                motion.append(0.0)
                roi_motion.append(0.0)
                prev_open = None
                prev_mouth_roi = None
                times.append(timestamp_ms)
                frame_idx += 1
                continue

            # 3) crop face from selected view
            face_crop_bgr, crop_box = crop_face_with_padding(
                selected_view,
                selected_detection.bounding_box,
                pad_ratio=0.35
            )

            if face_crop_bgr is None:
                visible.append(False)
                mouth_open.append(0.0)
                motion.append(0.0)
                roi_motion.append(0.0)
                prev_open = None
                prev_mouth_roi = None
                times.append(timestamp_ms)
                frame_idx += 1
                continue

            crop_h, crop_w = face_crop_bgr.shape[:2]

            # 4) run landmarker on face crop
            crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb = np.ascontiguousarray(crop_rgb)

            crop_mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=crop_rgb
            )

            lm_result = face_landmarker.detect_for_video(crop_mp_image, timestamp_ms)

            if not lm_result.face_landmarks:
                visible.append(False)
                mouth_open.append(0.0)
                motion.append(0.0)
                roi_motion.append(0.0)
                prev_open = None
                prev_mouth_roi = None
                times.append(timestamp_ms)
                frame_idx += 1
                continue

            # 5) mouth features
            face = lm_result.face_landmarks[0]

            upper = face[UPPER_LIP]
            lower = face[LOWER_LIP]
            left = face[LEFT_MOUTH]
            right = face[RIGHT_MOUTH]

            lip_gap = dist(upper, lower)
            mouth_width = dist(left, right)

            open_ratio = lip_gap / max(mouth_width, 1e-6)

            visible.append(True)
            mouth_open.append(open_ratio)

            if prev_open is None:
                motion.append(0.0)
            else:
                motion.append(abs(open_ratio - prev_open))
            prev_open = open_ratio

            curr_mouth_roi = crop_mouth_roi(face_crop_bgr, face, crop_w, crop_h)
            roi_score = roi_motion_score(prev_mouth_roi, curr_mouth_roi)
            roi_motion.append(roi_score)
            prev_mouth_roi = curr_mouth_roi

            times.append(timestamp_ms)
            frame_idx += 1

        cap.release()

        mouth_open = moving_average(mouth_open, k=5)
        motion = moving_average(motion, k=7)
        roi_motion = moving_average(roi_motion, k=7)

        activity = (
            0.2 * np.array(mouth_open)
            + 0.4 * np.array(motion)
            + 0.4 * np.array(roi_motion)
        )

        speaking_flags = []
        silent_flags = []

        ROI_MOTION_FLOOR = 0.008

        for v, score, roi_score in zip(visible, activity, roi_motion):
            if not v:
                speaking_flags.append(False)
                silent_flags.append(False)
                continue

            if roi_score < ROI_MOTION_FLOOR:
                speaking_flags.append(False)
                if score < 0.01:
                    silent_flags.append(True)
                else:
                    silent_flags.append(False)
                continue

            if score > 0.025:
                speaking_flags.append(True)
                silent_flags.append(False)
            elif score < 0.01:
                speaking_flags.append(False)
                silent_flags.append(True)
            else:
                speaking_flags.append(False)
                silent_flags.append(False)

        speaking_windows = build_windows(times, speaking_flags, min_len_ms=300)
        silent_windows = build_windows(times, silent_flags, min_len_ms=300)

        speaking_windows = merge_windows(speaking_windows, max_gap_ms=800)
        silent_windows = merge_windows(silent_windows, max_gap_ms=300)
        subtitle_visibility_analysis = analyze_subs_simple(subtitles_file, speaking_windows, silent_windows, tolerance_ms=150)
        return subtitle_visibility_analysis
    finally:
        cap.release()
        try:
            face_detector.close()
        except Exception:
            pass
        try:
            face_landmarker.close()
        except Exception:
            pass


def overlap_ms(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def analyze_subs_simple(subtitles_file, speaking_windows, silent_windows, tolerance_ms=150):
    """
    subs: list of dicts with:
      {
        "index": ...,
        "start_ms": ...,
        "end_ms": ...,
        "text": ...
      }

    Returns one compact analysis record per subtitle.
    """

    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    result = []

    for sub in subs:
        sub_start = int(sub.start.total_seconds() * 1000)
        sub_end = int(sub.end.total_seconds() * 1000)

        # 1) visible speaking during this subtitle?
        speaking_parts = []

        for win in speaking_windows:
            ov = overlap_ms(sub_start, sub_end, win["start_ms"], win["end_ms"])
            if ov > 0:
                speaking_parts.append({
                    "start_ms": max(sub_start, win["start_ms"]),
                    "end_ms": min(sub_end, win["end_ms"]),
                    "window": win,
                })

        has_visible_speaking = len(speaking_parts) > 0

        if has_visible_speaking:
            speaking_start_ms = min(p["start_ms"] for p in speaking_parts)
            speaking_end_ms = max(p["end_ms"] for p in speaking_parts)
        else:
            speaking_start_ms = None
            speaking_end_ms = None

        # 2) speaking starts in the middle of subtitle?
        speaking_starts_mid_sub = (
                speaking_start_ms is not None and speaking_start_ms > sub_start + tolerance_ms
        )
        speaking_starts_at_ms = speaking_start_ms if speaking_starts_mid_sub else None

        # 3) speaking disappears in the middle of subtitle?
        speaking_disappears_mid_sub = (
            speaking_end_ms is not None and speaking_end_ms < sub_end - tolerance_ms
        )
        speaking_disappears_at_ms = speaking_end_ms if speaking_disappears_mid_sub else None

        # 3) first visible silence after subtitle
        silence_after_sub_start_ms = None
        for win in silent_windows:
            if win["start_ms"] >= sub_end:
                silence_after_sub_start_ms = win["start_ms"]
                break

        silence_after_sub = silence_after_sub_start_ms is not None

        result.append({
            "index": sub.index,
            "has_visible_speaking": has_visible_speaking,
            "speaking_start_ms": speaking_start_ms,
            "speaking_end_ms": speaking_end_ms,
            "speaking_starts_mid_sub": speaking_starts_mid_sub,
            "speaking_starts_at_ms": speaking_starts_at_ms,
            "speaking_disappears_mid_sub": speaking_disappears_mid_sub,
            "speaking_disappears_at_ms": speaking_disappears_at_ms,

            "silence_after_sub": silence_after_sub,
            "silence_after_sub_start_ms": silence_after_sub_start_ms
        })


    return result