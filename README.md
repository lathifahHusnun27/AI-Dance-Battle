# AI Dance Battle

AI Dance Battle adalah permainan interaktif dua pemain berbasis computer vision yang menggunakan YOLOv8-Pose untuk mendeteksi pose tubuh secara real-time melalui webcam.

Setiap pemain diminta mengikuti target pose yang ditampilkan. Sistem akan membandingkan keypoint tubuh pemain dengan target pose, menghitung tingkat kemiripan, memberikan feedback, dan menentukan pemenang berdasarkan total skor.

## Fitur

- Deteksi pose tubuh secara real-time menggunakan YOLOv8-Pose
- Mendukung dua pemain secara bersamaan
- Visualisasi skeleton dan keypoint tubuh
- Normalisasi keypoint berdasarkan posisi tubuh
- Perhitungan kemiripan pose pemain dengan target
- Sistem skor dan feedback:
  - PERFECT
  - GREAT
  - OK
  - MISS
  - FAIL
- Countdown untuk setiap pose
- Pemilihan dan pemutaran musik
- Tampilan target pose
- Penentuan pemenang berdasarkan total skor
- Fitur restart dan keluar dari permainan

## Teknologi yang Digunakan

- Python
- Ultralytics YOLOv8-Pose
- OpenCV
- NumPy
- Pygame

## Struktur Repository

```text
AI-Dance-Battle/
├── main.py
├── pose.py
├── score.py
├── requirements.txt
├── music/
└── README.md
