from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/processed/mfcc/mfcc_dataset.npz'
MODEL_PATH = ROOT / 'models/mfcc_cnn1d_mild.keras'
BEST_PATH = ROOT / 'models/mfcc_cnn1d_mild_best.keras'
NORM_PATH = ROOT / 'models/mfcc_cnn1d_mild_normalization.npz'
SEED = 42
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 3

def build_model():
    reg = regularizers.l2(0.0001)
    model = keras.Sequential([keras.Input(shape=(87, 40), name='mfcc_sequence'), layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False, kernel_regularizer=reg), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPooling1D(pool_size=2), layers.Conv1D(64, kernel_size=5, padding='same', use_bias=False, kernel_regularizer=reg), layers.BatchNormalization(), layers.Activation('relu'), layers.GlobalAveragePooling1D(), layers.Dropout(0.3), layers.Dense(32, activation='relu', kernel_regularizer=reg), layers.Dropout(0.2), layers.Dense(NUM_CLASSES, activation='softmax')], name='mfcc_cnn1d')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    tf.keras.utils.set_random_seed(SEED)
    with np.load(DATA_PATH) as data:
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int64)
        splits = data['splits']
    train_mask = splits == 'train'
    val_mask = splits == 'validation'
    X_train, y_train = (X[train_mask], y[train_mask])
    X_val, y_val = (X[val_mask], y[val_mask])
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError('Training or validation split is empty.')
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), 1e-06)
    np.savez(NORM_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))
    X_train = ((X_train - mean) / std).transpose(0, 2, 1)
    X_val = ((X_val - mean) / std).transpose(0, 2, 1)
    model = build_model()
    model.summary()
    callbacks = [keras.callbacks.ModelCheckpoint(BEST_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-05, verbose=1)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=callbacks, verbose=1, class_weight={0: 0.875, 1: 1.25, 2: 1.0})
    model.save(MODEL_PATH)
    best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
    best_val_accuracy = float(max(history.history['val_accuracy']))
    print('\nTraining complete.')
    print(f'Best validation epoch: {best_epoch}')
    print(f'Best validation accuracy: {best_val_accuracy:.4f}')
    print(f'Best model: {BEST_PATH}')
    print(f'Final model: {MODEL_PATH}')
    print(f'Normalization: {NORM_PATH}')
if __name__ == '__main__':
    main()
