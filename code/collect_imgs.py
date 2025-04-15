import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Modelo Holístico
mp_drawing = mp.solutions.drawing_utils  # Propriedades dos desenhos
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB
    image.flags.writeable = False  # Não pode mais ser editada
    results = model.process(image)  # Faz previsão e identificação das imagens
    image.flags.writeable = True  # Pode ser editada de novo
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte de novo de RGB para BGR
    return image, results  # Traz os resultados

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS) #Desenha landmarks do rosto
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Desenha landmarkls de pose
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #desenha landmarks da mão esquerda
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Desenha landmarks da mão direita

def draw_styled_landmarks(image, results):
    # Desenha landmarks do rosto
    mp_drawing.draw_landmarks(image, results.face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #Muda os atributos dos landmarks
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) #muda os atributos das conexões
    # Desenha landmarkls de pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),# Muda os atributos dos landmarks
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1,circle_radius=1))# muda os atributos das conexões
    # desenha landmarks da mão esquerda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),# Muda os atributos dos landmarks
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1,circle_radius=1))# muda os atributos das conexões
    # Desenha landmarks da mão direita
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),# Muda os atributos dos landmarks
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1,circle_radius=1))# muda os atributos das conexões

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Procura a câmera
        ret, frame = cap.read()

        # Indentificação das imagens
        image, results = mediapipe_detection(frame, holistic)

        # Exibe a câmera se ela for encontrada
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)
        # Quebra o loop se apertar 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
