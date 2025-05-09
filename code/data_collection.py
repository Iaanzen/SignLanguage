import os
import cv2
import numpy as np
import mediapipe as mp

from tqdm import tqdm

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Função para extrair keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Caminhos
VIDEOS_DIR = 'VIDEOS_POR_PALAVRA'
KEYPOINTS_DIR = 'KEYPOINTS'

os.makedirs(KEYPOINTS_DIR, exist_ok=True)

# Inicia o modelo
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for classe in tqdm(os.listdir(VIDEOS_DIR), desc="Classes"):
        classe_path = os.path.join(VIDEOS_DIR, classe)
        if not os.path.isdir(classe_path):
            continue

        for idx, video_file in enumerate(os.listdir(classe_path)):
            video_path = os.path.join(classe_path, video_file)

            cap = cv2.VideoCapture(video_path)
            keypoints_sequence = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True

                keypoints = extract_keypoints(results)
                keypoints_sequence.append(keypoints)

            cap.release()

            # Salva os keypoints como npy
            dest_folder = os.path.join(KEYPOINTS_DIR, classe, f"video_{idx}")
            os.makedirs(dest_folder, exist_ok=True)

            for frame_num, keypoints in enumerate(keypoints_sequence):
                np.save(os.path.join(dest_folder, f"{frame_num}.npy"), keypoints)
