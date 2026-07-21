import numpy as np

def normalize_keypoints(keypoints_raw: np.ndarray) -> dict:
    """
    Konversi output YOLOv8 keypoints (shape: [17, 3]) ke dict {idx: (x_norm, y_norm)}.
    Koordinat dinormalisasi terhadap bounding box tubuh (shoulder-hip span).
    Keypoint dengan confidence < threshold diabaikan.
    """
    CONF_THRESHOLD = 0.4
    result = {}
    valid_pts = []

    for i, kp in enumerate(keypoints_raw):
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf >= CONF_THRESHOLD:
            result[i] = (x, y)
            valid_pts.append((x, y))

    if len(valid_pts) < 4:
        return {}

    # Normalisasi: scale relatif terhadap rentang bounding box keypoints
    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    span_x = x_max - x_min if x_max != x_min else 1
    span_y = y_max - y_min if y_max != y_min else 1

    normalized = {}
    for idx, (x, y) in result.items():
        nx = (x - x_min) / span_x
        ny = (y - y_min) / span_y
        normalized[idx] = (nx, ny)

    return normalized


def compute_similarity(player_kps: dict, target_kps: dict) -> float:
    """
    Hitung similarity score antara pose pemain dan target pose.
    Metode: Mean Squared Error (MSE) per keypoint, dikonversi ke skor 0-100.
    Hanya keypoints yang ada di target_kps yang dievaluasi.
    
    Returns:
        float: Skor 0.0 - 100.0
    """
    if not player_kps or not target_kps:
        return 0.0

    errors = []
    for idx, (tx, ty) in target_kps.items():
        if idx in player_kps:
            px, py = player_kps[idx]
            dist = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)
            errors.append(dist)
        else:
            # Keypoint tidak terdeteksi → penalti maksimal
            errors.append(1.0)

    if not errors:
        return 0.0

    mse = np.mean(errors)
    # Mapping yang jauh lebih longgar: mse=0 → 100, mse>=0.8 → 0 (sangat forgiving)
    score = max(0.0, 100.0 * (1.0 - mse / 0.8))
    return round(score, 1)


def get_feedback(score: float) -> tuple[str, tuple]:
    """
    Feedback teks dan warna berdasarkan skor.
    Returns: (label, BGR_color)
    """
    if score >= 90:
        return "PERFECT!", (0, 255, 150)
    elif score >= 70:
        return "GREAT!", (0, 200, 255)
    elif score >= 50:
        return "OK", (0, 165, 255)
    elif score >= 30:
        return "MISS", (0, 80, 255)
    else:
        return "FAIL", (0, 0, 220)