import sqlite3
import pickle
import numpy as np
import cv2
import mediapipe as mp

# Conexão com o banco
conn = sqlite3.connect('gestos.db')
cursor = conn.cursor()

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

# Pega todas as imagens
cursor.execute('SELECT classe, nome_arquivo, imagem FROM imagens')
registros = cursor.fetchall()

for classe, nome_arquivo, imagem_blob in registros:
    # Decodifica os bytes em imagem
    nparr = np.frombuffer(imagem_blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[!] Erro ao ler imagem {nome_arquivo}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:  # 21 landmarks x 2 (x e y)
                data.append(data_aux)
                labels.append(str(classe))
            else:
                print(f"[!] Imagem ignorada (vetor incompleto): {nome_arquivo}")

conn.close()

# Salva os dados no arquivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ {len(data)} amostras processadas e salvas no arquivo data.pickle")
