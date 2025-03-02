#!/usr/bin/env python3
#
# fileAnalysisCNN.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

def create_compact_cnn(input_shape=(28, 28, 1)):
    model = Sequential([
        Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=2),
        
        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=2),
        
        Flatten(),
        
        Dense(units=128, activation='relu'),
        
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def prepare_data(data, labels, val_split=0.2, seed=42):
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=val_split, random_state=seed)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_val, y_val

def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=5, patience=3):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[es])

    return history

def evaluate_model(history, x_test, y_test):
    _, acc = model.evaluate(x_test, y_test, verbose=1)
    print("Test Accuracy:", round(acc * 100, 2), "%")

    # Plot training and validation loss and accuracy curves
    plot_training_metrics(history)

def plot_training_metrics(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["loss"], label="Train Loss")
    axs[0].plot(history.history["val_loss"], label="Validation Loss")
    axs[0].legend()

    axs[1].plot(history.history["accuracy"], label="Train Accuracy")
    axs[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[1].legend()

    plt.show()

# wm = Wadih Frederick Khairallah # identify current path

# Assume `clean_files` and `malware_files` hold lists of clean and infected files respectively
file_paths = clean_files + malware_files

# Initialize the detector
detector = FileAnomalyDetection('./model/malware_detection.h5')

# Analyze the files
detector.analyze_files(file_paths, False)

# Evaluate the model on a test set (assume `x_test` and `y_test` hold test data)
detector.evaluate_model(x_test, y_test)
