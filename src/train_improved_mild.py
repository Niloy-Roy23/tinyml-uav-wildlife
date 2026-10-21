import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
data = np.load('data/processed/windowed_dataset.npz')
X = data['X']
y = data['y']
splits = data['splits']
X_train = X[splits == 'train']
y_train = y[splits == 'train']
X_val = X[splits == 'validation']
y_val = y[splits == 'validation']
X_test = X[splits == 'test']
y_test = y[splits == 'test']
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print('Training data:', X_train.shape)
print('Validation data:', X_val.shape)
print('Test data:', X_test.shape)
model = keras.Sequential([layers.Input(shape=(40, 87, 1)), layers.Conv2D(16, (3, 3), activation='relu', padding='same'), layers.MaxPooling2D((2, 2)), layers.Conv2D(32, (3, 3), activation='relu', padding='same'), layers.MaxPooling2D((2, 2)), layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.GlobalAveragePooling2D(), layers.Dense(32, activation='relu'), layers.Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
import os
import json
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(filepath='models/improved_cnn_mild.keras', monitor='val_loss', save_best_only=True, verbose=1), keras.callbacks.CSVLogger('reports/improved_mild_training_history.csv')]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=callbacks, class_weight={0: 0.875, 1: 1.25, 2: 1.0})
model.save('models/improved_cnn_mild_final.keras')
with open('reports/improved_mild_training_history.json', 'w') as f:
    json.dump({key: [float(value) for value in values] for key, values in history.history.items()}, f, indent=2)
print('Best model saved to models/improved_cnn_mild.keras')
print('Training history saved to reports/')
