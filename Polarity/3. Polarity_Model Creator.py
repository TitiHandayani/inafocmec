import tensorflow
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, Callback
from obspy import UTCDateTime, read
import obspy
import os
from tqdm import tqdm
import pandas as pd
import pickle
# Load the data

def load_var(model_name:str):
    model_path = f'./var/{model_name}'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
            print(f"Model '{model_name}' loaded successfully.")
            return loaded_model
    else:
        print(f"Model '{model_name}' does not exist.")

# Training Data
X_train = load_var("X_train.pkl")
Y_train = load_var("Y_train.pkl")
print(X_train.shape, Y_train.shape)
print(X_train.dtype, Y_train.dtype)

# Validation Data
X_validation = load_var("X_validation.pkl")
Y_validation = load_var("Y_validation.pkl")
print(X_validation.shape, Y_validation.shape)
print(X_validation.dtype, Y_validation.dtype)

# Test Data
X_test = load_var("X_test.pkl")
Y_test = load_var("Y_test.pkl")
print(X_test.shape, Y_test.shape)
print(X_test.dtype, Y_test.dtype)


print([np.isnan(X_train).any(), 
       np.isnan(Y_train).any(), 
       np.isnan(X_validation).any(), 
       np.isnan(Y_validation).any(), 
       np.isnan(X_test).any(), 
       np.isnan(Y_test).any()]
      )  # Check for NaNs in all variables
print([np.isinf(X_train).any(), 
       np.isinf(Y_train).any(), 
       np.isinf(X_validation).any(), 
       np.isinf(Y_validation).any(), 
       np.isinf(X_test).any(), 
       np.isinf(Y_test).any()]
      )  # Check for infinite values in all variables

def createNewModel(input_shape):
    input_layer = layers.Input(shape=input_shape)
    # Block 1
    x1 = layers.Conv1D(128, 3, activation='relu', padding='same')(input_layer)
    x1 = layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling1D(pool_size=1)(x1)
    x1 = layers.Dropout(0.5)(x1)

    # Block 2
    x2 = layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
    x2 = layers.Conv1D(128, 3, activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling1D(pool_size=1)(x2)
    x2 = layers.Dropout(0.5)(x2)

    # Block 3
    x3 = layers.Conv1D(128, 3, activation='relu', padding='same')(x2)
    x3 = layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
    x3 = layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
    x3 = layers.MaxPooling1D(pool_size=1)(x3)
    x3 = layers.Dropout(0.5)(x3)

    # Block 4
    x4 = layers.Conv1D(128, 3, activation='relu', padding='same')(x3)
    x4 = layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
    x4 = layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
    x4 = layers.MaxPooling1D(pool_size=1)(x4)
    x4 = layers.Dropout(0.5)(x4)

    # Block 5
    x5 = layers.Conv1D(128, 3, activation='relu', padding='same')(x4)
    x5 = layers.Conv1D(128, 3, activation='relu', padding='same')(x5)
    x5 = layers.Conv1D(128, 3, activation='relu', padding='same')(x5)
    x5 = layers.MaxPooling1D(pool_size=1)(x5)
    # No pooling here, keeping the shape as is
    x5 = layers.Dropout(0.5)(x5) 
    # Concatenate outputs from blocks
    x = layers.Concatenate()([x3, x4, x5])
    # Flatten the concatenated output before passing to Dense layers
    x = layers.Flatten()(x)
    # Output layer with the number of classes
    output_layer = layers.Dense(3, activation='softmax')(x)
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Display the model summary
    model.summary()
    return model

# Custom callback to enable early stopping after a specific number of epochs
class EarlyStoppingAfterEpoch(Callback):
    def __init__(self, start_epoch, patience=3, min_delta=0, monitor='val_loss'):
        super(EarlyStoppingAfterEpoch, self).__init__()
        self.start_epoch = start_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Only apply early stopping after the start_epoch
        if epoch >= self.start_epoch:
            current = logs.get(self.monitor)
            if self.best is None or current < self.best - self.min_delta:
                self.best = current
                self.wait = 0  # Reset patience counter if improvement
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print(f'\nEpoch {epoch + 1}: Early stopping triggered after waiting for {self.patience} epochs without improvement.')

# Example usage


# Define different values for epochs and batch_size
epoch_values = [30, 100, 200]  # Example epoch options
# epoch_values = [30, 80, 100, 200]  # Example epoch options
batch_size_values = [64,300,1000, 3000]  # Example batch size options

# Set up EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Or 'val_accuracy'
    patience=5,          # Number of epochs to wait for improvement
    restore_best_weights=True
)

# Iterate over each combination of epoch and batch size
for epochs in epoch_values:
    for batch_size in batch_size_values:
        early_stopping_custom = EarlyStoppingAfterEpoch(start_epoch=epochs/5*4, patience=3)  # Enable after 10th epoch
        print(f"Training with epochs={epochs} and batch_size={batch_size}")
        # Create a fresh model
        input_shape = (X_train.shape[1], 1)
        model = createNewModel(input_shape)

        # Train the model with early stopping
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_validation, Y_validation),
            callbacks=[early_stopping_custom]
        )
        stopped_epoch = early_stopping_custom.stopped_epoch if early_stopping_custom.stopped_epoch else epochs

        # Save the training history to a pickle file
        history_filename = f'./model/model2_ep{stopped_epoch}-{epochs}_bs{batch_size}_hist.pkl'
        with open(history_filename, 'wb') as file:
            pickle.dump(history.history, file)
        
        # Save the model
        model_filename = f'./model/model2_ep{stopped_epoch}-{epochs}_bs{batch_size}.h5'
        model.save(model_filename)
        
        print(f"Saved history to {history_filename} and model to {model_filename}")
