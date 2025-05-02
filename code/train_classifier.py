import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

# Caminho onde estão os keypoints
KEYPOINTS_DIR = 'KEYPOINTS'

# Lista de classes com base nas pastas
classes = sorted(os.listdir(KEYPOINTS_DIR))
label_map = {label: idx for idx, label in enumerate(classes)}

sequences, labels = [], []

for label in classes:
    label_path = os.path.join(KEYPOINTS_DIR, label)
    for sequence_folder in os.listdir(label_path):
        seq_path = os.path.join(label_path, sequence_folder)
        frames = sorted(os.listdir(seq_path), key=lambda x: int(x.split('.')[0]))

        window = []
        for frame_name in frames:
            frame_path = os.path.join(seq_path, frame_name)
            keypoints = np.load(frame_path)
            window.append(keypoints)

        if len(window) >= 30:
            sequences.append(window[:30])  # pega os 30 primeiros
            labels.append(label_map[label])
        else:
            print(f"❌ Ignorando {seq_path} (só tem {len(window)} frames)")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Salvar os dados (opcional, se quiser reutilizar depois)
with open('code/data.pickle', 'wb') as f:
    pickle.dump((X, y, classes), f)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=labels, random_state=42)

print(f"Classes encontradas: {classes}")
print(f"Shape X: {X.shape}, y: {y.shape}")
