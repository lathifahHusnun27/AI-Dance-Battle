"""
AI Dance Battle Game
Platform : Desktop Python
AI Model : YOLOv8-pose (ultralytics)
UI/Audio : OpenCV overlay + pygame.mixer
"""

import cv2
import numpy as np
import pygame
import time
import os
import sys
from ultralytics import YOLO

from score import normalize_keypoints, compute_similarity, get_feedback
from pose import POSES

# ─── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_W, FRAME_H = 1280, 720
MODEL_PATH     = "yolov8n-pose.pt"   # auto-download jika belum ada
MUSIC_DIR      = "music"
COUNTDOWN_SEC  = 3                   # hitung mundur sebelum pose dinilai
SCORE_HOLD_SEC = 0.5                 # interval sampling skor saat hold pose
# ───────────────────────────────────────────────────────────────────────────────


def list_songs(music_dir: str) -> list[str]:
    if not os.path.isdir(music_dir):
        os.makedirs(music_dir)
        return []
    return [f for f in os.listdir(music_dir) if f.endswith((".mp3", ".wav", ".ogg"))]


def select_music_menu(songs: list[str]) -> str | None:
    """Menu pilih lagu via terminal sebelum game dimulai."""
    if not songs:
        print("[INFO] Tidak ada file musik di folder 'music/'. Game berjalan tanpa musik.")
        return None
    print("\n=== Pilih Lagu ===")
    for i, s in enumerate(songs):
        print(f"  {i + 1}. {s}")
    print("  0. Tanpa musik")
    while True:
        try:
            choice = int(input("Pilihan: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(songs):
                return os.path.join(MUSIC_DIR, songs[choice - 1])
        except ValueError:
            pass
        print("Input tidak valid, coba lagi.")


def draw_skeleton(frame: np.ndarray, keypoints_raw: np.ndarray, pt_color=(0, 255, 200), line_color=(200, 255, 100), conf_thr: float = 0.4):
    """Gambar skeleton YOLOv8-pose (COCO 17 keypoints) di atas frame."""
    COCO_EDGES = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    pts = {}
    for i, kp in enumerate(keypoints_raw):
        x, y, conf = int(kp[0]), int(kp[1]), float(kp[2])
        if conf >= conf_thr:
            pts[i] = (x, y)
            cv2.circle(frame, (x, y), 5, pt_color, -1)

    for a, b in COCO_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], line_color, 2)


def draw_target_pose_overlay(frame: np.ndarray, target_kps: dict, h: int, w: int):
    """
    Render target pose sebagai silhouette titik-titik di pojok kanan atas.
    Koordinat ternormalisasi di-remap ke panel kecil 200x200.
    """
    PANEL = 150
    PAD   = 10
    COCO_EDGES = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    # Pindah ke tengah atas (di bawah hitung mundur)
    ox = w // 2 - PANEL // 2
    oy = 90

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + PANEL, oy + PANEL), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    pts = {}
    for idx, (nx, ny) in target_kps.items():
        px = int(ox + nx * PANEL)
        py = int(oy + ny * PANEL)
        pts[idx] = (px, py)
        cv2.circle(frame, (px, py), 5, (255, 220, 0), -1)

    for a, b in COCO_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (200, 180, 0), 2)

    cv2.putText(frame, "TARGET", (ox + 5, oy + PANEL - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_hud(frame, pose_name, countdown, score_p1, feedback_p1, color_p1,
             score_p2, feedback_p2, color_p2,
             total_score_p1, total_score_p2, pose_idx, total_poses, h, w):
    """Render HUD untuk 2 Player di atas frame."""
    # Pose name & Countdown (Tengah)
    cv2.putText(frame, f"Pose: {pose_name}", (w // 2 - 100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    bar_w = int((countdown / COUNTDOWN_SEC) * 300)
    bar_x = w // 2 - 150
    cv2.rectangle(frame, (bar_x, 60), (bar_x + 300, 80), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, 60), (bar_x + bar_w, 80), (0, 220, 120), -1)
    cv2.putText(frame, f"{countdown:.1f}s", (bar_x + 310, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Player 1 (Kiri)
    cv2.putText(frame, "PLAYER 1", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 150, 0), 2)
    cv2.putText(frame, f"Match: {score_p1:.0f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if feedback_p1:
        cv2.putText(frame, feedback_p1, (20, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, color_p1, 2)
    cv2.putText(frame, f"Score: {total_score_p1:.0f}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)

    # Player 2 (Kanan)
    cv2.putText(frame, "PLAYER 2", (w - 180, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 50, 255), 2)
    cv2.putText(frame, f"Match: {score_p2:.0f}%", (w - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if feedback_p2:
        cv2.putText(frame, feedback_p2, (w - 180, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, color_p2, 2)
    cv2.putText(frame, f"Score: {total_score_p2:.0f}", (w - 180, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 50, 255), 2)

    # Progress pose (Kanan Bawah)
    cv2.putText(frame, f"Pose {pose_idx + 1}/{total_poses}", (w - 180, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def result_screen(cap, total_score_p1, total_score_p2, total_poses):
    """Tampilkan layar hasil akhir dan pengumuman pemenang."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Gelap overlay
        overlay = np.zeros_like(frame)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        h, w = frame.shape[:2]

        if total_score_p1 > total_score_p2:
            winner_text = "PLAYER 1 WINS!"
            win_color = (255, 150, 0)
        elif total_score_p2 > total_score_p1:
            winner_text = "PLAYER 2 WINS!"
            win_color = (200, 50, 255)
        else:
            winner_text = "IT'S A TIE!"
            win_color = (200, 200, 200)

        cv2.putText(frame, "GAME OVER", (w // 2 - 160, h // 2 - 120),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 220, 0), 3)
        
        cv2.putText(frame, winner_text, (w // 2 - 180, h // 2 - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, win_color, 3)

        cv2.putText(frame, f"Player 1: {total_score_p1:.0f}", (w // 2 - 200, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 0), 2)
        cv2.putText(frame, f"Player 2: {total_score_p2:.0f}", (w // 2 + 50, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 50, 255), 2)

        cv2.putText(frame, "Press Q to quit | R to restart", (w // 2 - 180, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        cv2.imshow("AI Dance Battle", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('r'):
            return True
    return False


def run_game(model: YOLO, cap: cv2.VideoCapture, song_path: str | None):
    pygame.mixer.init()
    if song_path and os.path.isfile(song_path):
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play(-1)  # loop

    total_score_p1 = 0.0
    total_score_p2 = 0.0
    pose_idx = 0

    while pose_idx < len(POSES):
        pose_data   = POSES[pose_idx]
        pose_name   = pose_data["name"]
        target_kps  = pose_data["keypoints"]
        hold_sec    = pose_data["duration"]

        countdown   = hold_sec
        last_time   = time.time()
        
        current_score_p1 = 0.0
        current_score_p2 = 0.0
        feedback_p1 = ""
        feedback_p2 = ""
        color_p1 = (255, 255, 255)
        color_p2 = (255, 255, 255)
        
        score_samples_p1 = []
        score_samples_p2 = []

        while countdown > 0:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            h, w  = frame.shape[:2]

            # YOLOv8 inference
            results = model(frame, verbose=False)[0]

            player_kps_norm_p1 = {}
            player_kps_norm_p2 = {}

            if results.keypoints is not None and len(results.keypoints.data) > 0:
                # Ambil semua person yang terdeteksi
                people = []
                for kp_data in results.keypoints.data:
                    kp_np = kp_data.cpu().numpy()
                    # Hitung rata-rata X untuk person ini (untuk mengetahui kiri/kanan)
                    xs = [p[0] for p in kp_np if p[2] > 0.4]
                    if xs:
                        avg_x = np.mean(xs)
                        people.append((avg_x, kp_np))
                
                # Urutkan berdasarkan X (Kiri -> P1, Kanan -> P2)
                people.sort(key=lambda x: x[0])
                
                if len(people) >= 1:
                    kp_p1 = people[0][1]
                    draw_skeleton(frame, kp_p1, pt_color=(255, 150, 0), line_color=(200, 100, 0))
                    player_kps_norm_p1 = normalize_keypoints(kp_p1)
                
                if len(people) >= 2:
                    kp_p2 = people[1][1]
                    draw_skeleton(frame, kp_p2, pt_color=(200, 50, 255), line_color=(150, 0, 200))
                    player_kps_norm_p2 = normalize_keypoints(kp_p2)

            current_score_p1 = compute_similarity(player_kps_norm_p1, target_kps)
            score_samples_p1.append(current_score_p1)
            feedback_p1, color_p1 = get_feedback(current_score_p1)

            current_score_p2 = compute_similarity(player_kps_norm_p2, target_kps)
            score_samples_p2.append(current_score_p2)
            feedback_p2, color_p2 = get_feedback(current_score_p2)

            # Update countdown
            now = time.time()
            dt  = now - last_time
            last_time = now
            countdown = max(0.0, countdown - dt)

            # Overlay
            draw_target_pose_overlay(frame, target_kps, h, w)
            draw_hud(frame, pose_name, countdown, 
                     current_score_p1, feedback_p1, color_p1,
                     current_score_p2, feedback_p2, color_p2,
                     total_score_p1, total_score_p2, pose_idx, len(POSES), h, w)

            cv2.imshow("AI Dance Battle", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pygame.mixer.music.stop()
                return total_score_p1, total_score_p2

        # Skor pose = rata-rata sample saat hold
        if score_samples_p1:
            total_score_p1 += np.mean(score_samples_p1[-int(hold_sec / SCORE_HOLD_SEC):] or score_samples_p1)
        if score_samples_p2:
            total_score_p2 += np.mean(score_samples_p2[-int(hold_sec / SCORE_HOLD_SEC):] or score_samples_p2)

        pose_idx += 1
        time.sleep(0.5)  # jeda singkat antar pose

    pygame.mixer.music.stop()
    return total_score_p1, total_score_p2


def main():
    print("=== AI Dance Battle Game ===")
    songs = list_songs(MUSIC_DIR)
    song_path = select_music_menu(songs)

    print("[INFO] Loading YOLOv8-pose model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi.")
        sys.exit(1)

    print("[INFO] Game dimulai! Tekan Q untuk keluar.")

    while True:
        total_score_p1, total_score_p2 = run_game(model, cap, song_path)
        restart = result_screen(cap, total_score_p1, total_score_p2, len(POSES))
        if not restart:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()