import os

from keras.src.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam

from code.create_dataset import custom_actions
from code.train_classifier import X_train, y_train, X_test, y_test

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True
)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 1662)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
#
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stopping])

print(model.summary())