# YOLOv8-pose COCO 17 keypoints index:
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
#
# Koordinat (x, y) ternormalisasi 0.0–1.0 relatif terhadap bounding box tubuh.
# x: 0=kiri, 1=kanan | y: 0=atas, 1=bawah
# Hanya keypoints relevan per pose yang dievaluasi.

POSES = [
    {
        "name": "T-Pose",
        "duration": 3,
        "keypoints": {
            5:  (0.30, 0.35),  # left shoulder
            6:  (0.70, 0.35),  # right shoulder
            7:  (0.10, 0.35),  # left elbow — tangan lurus ke samping
            8:  (0.90, 0.35),  # right elbow
            9:  (0.00, 0.35),  # left wrist
            10: (1.00, 0.35),  # right wrist
            11: (0.40, 0.60),  # left hip
            12: (0.60, 0.60),  # right hip
            13: (0.40, 0.80),  # left knee lurus
            14: (0.60, 0.80),  # right knee lurus
            15: (0.40, 1.00),  # left ankle
            16: (0.60, 1.00),  # right ankle
        }
    },
    {
        "name": "Victory V",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.20, 0.18),  # kedua elbow diagonal ke atas
            8:  (0.80, 0.18),
            9:  (0.10, 0.05),  # kedua wrist tinggi membentuk V
            10: (0.90, 0.05),
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.35, 0.80),  # Kaki sedikit mengangkang (V shape terbalik)
            14: (0.65, 0.80),
            15: (0.30, 1.00),
            16: (0.70, 1.00),
        }
    },
    {
        "name": "Left Punch",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.12, 0.35),  # left elbow maju
            8:  (0.72, 0.40),  # right elbow ditekuk di pinggang
            9:  (0.02, 0.35),  # left wrist maju (punch)
            10: (0.65, 0.48),  # right wrist di pinggang
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.30, 0.75),  # Kuda-kuda kiri: lutut kiri ditekuk ke depan
            14: (0.60, 0.80),  # Lutut kanan lurus
            15: (0.25, 1.00),
            16: (0.60, 1.00),
        }
    },
    {
        "name": "Hands Up",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.30, 0.12),  # kedua elbow ke atas
            8:  (0.70, 0.12),
            9:  (0.25, 0.02),  # kedua wrist setinggi mungkin
            10: (0.75, 0.02),
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.40, 0.80),  # Kaki lurus
            14: (0.60, 0.80),
            15: (0.40, 1.00),
            16: (0.60, 1.00),
        }
    },
    {
        "name": "Right Punch",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.28, 0.40),  # left elbow ditekuk di pinggang
            8:  (0.88, 0.35),  # right elbow maju
            9:  (0.35, 0.48),  # left wrist di pinggang
            10: (0.98, 0.35),  # right wrist maju (punch)
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.40, 0.80),  # Lutut kiri lurus
            14: (0.70, 0.75),  # Kuda-kuda kanan: lutut kanan ditekuk ke depan
            15: (0.40, 1.00),
            16: (0.75, 1.00),
        }
    },
    {
        "name": "Left Hand Up",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.22, 0.12),  # left elbow ke atas
            8:  (0.72, 0.42),  # right elbow di bawah (rileks)
            9:  (0.18, 0.02),  # left wrist tinggi
            10: (0.68, 0.55),  # right wrist ke bawah
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.40, 0.80),
            14: (0.60, 0.80),
            15: (0.40, 1.00),
            16: (0.60, 1.00),
        }
    },
    {
        "name": "Cross Arms",
        "duration": 3,
        "keypoints": {
            5:  (0.35, 0.35),
            6:  (0.65, 0.35),
            7:  (0.55, 0.42),  # left elbow menyilang ke kanan
            8:  (0.45, 0.42),  # right elbow menyilang ke kiri
            9:  (0.65, 0.38),  # left wrist di kanan (silang)
            10: (0.35, 0.38),  # right wrist di kiri (silang)
            11: (0.40, 0.60),
            12: (0.60, 0.60),
            13: (0.40, 0.80),
            14: (0.60, 0.80),
            15: (0.40, 1.00),
            16: (0.60, 1.00),
        }
    },
    {
        "name": "Superhero",
        "duration": 4,
        "keypoints": {
            5:  (0.30, 0.32),
            6:  (0.70, 0.32),
            7:  (0.10, 0.20),  # left elbow diagonal atas-samping
            8:  (0.72, 0.42),  # right elbow ditekuk di pinggang
            9:  (0.02, 0.10),  # left wrist tinggi ke depan (superhero pose)
            10: (0.65, 0.50),  # right wrist di pinggang
            11: (0.38, 0.58),
            12: (0.62, 0.58),
            13: (0.30, 0.75),  # Kuda-kuda superhero (lutut ditekuk/merendah)
            14: (0.70, 0.75),
            15: (0.25, 0.95),  # Kaki agak melebar
            16: (0.75, 0.95),
        }
    },
]