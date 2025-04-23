import os
import cv2
import numpy as np
import time
import mediapipe as mp
from code.collect_imgs import mp_holistic, mediapipe_detection, extract_keypoints, draw_styled_landmarks
from code.create_dataset import actions, no_sequences, sequence_length, DATA_PATH

# Inicia captura da webcam
cap = cv2.VideoCapture(0)
stop = False  # Controle de parada geral

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                # Processa a imagem com MediaPipe
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Exibe mensagens na tela no primeiro frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Extrai e salva os keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Mostra a imagem processada
                cv2.imshow('OpenCV Feed', image)

                # Se pressionar 'q', interrompe
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
