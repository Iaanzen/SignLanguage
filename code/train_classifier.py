import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

# Carregar os dados
with open('code/data.pickle', 'rb') as f:
    X, y, classes = pickle.load(f)

print(f"✅ Dados carregados! Total de amostras: {X.shape[0]}")
print(f"Classes: {classes}")

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y.argmax(axis=1), random_state=42
)

# Criar o modelo
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar resumo
model.summary()

# Callback para salvar o melhor modelo
checkpoint = ModelCheckpoint(
    'code/signal_classifier.h5', monitor='val_accuracy', save_best_only=True, verbose=1
)

# Treinar
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

print("✅ Treinamento finalizado!")
