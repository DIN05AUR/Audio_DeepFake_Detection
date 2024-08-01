import joblib
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import os

# Global constant
max_length = 204

# Feature Extraction Process
def extract_features(audio_file, sr=16000):
    try:
        audio, _ = librosa.load(audio_file, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        # Padding or truncation for consistent shape
        if mfccs.shape[1] < max_length:
            pad_width = max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]

        # Flatten the third dimension
        mfccs_flat = mfccs.reshape(-1)

        return mfccs_flat
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Loading audio files from directory
def load_audio_files_from_directory(directory):
    audio_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.wav')]
    return audio_files

# Script used for training for reference process
def train_model(real_audio_directory, fake_audio_directory, val_real_audio_directory, val_fake_audio_directory):
    # Print available devices
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)

    real_audio_files = load_audio_files_from_directory(real_audio_directory)
    fake_audio_files = load_audio_files_from_directory(fake_audio_directory)
    val_real_audio_files = load_audio_files_from_directory(val_real_audio_directory)
    val_fake_audio_files = load_audio_files_from_directory(val_fake_audio_directory)

    # Extract features and handle exceptions
    real_features = [extract_features(audio_file, sr=16000) for audio_file in real_audio_files if extract_features(audio_file, sr=16000) is not None]
    fake_features = [extract_features(audio_file, sr=16000) for audio_file in fake_audio_files if extract_features(audio_file, sr=16000) is not None]
    val_real_features = [extract_features(audio_file, sr=16000) for audio_file in val_real_audio_files if extract_features(audio_file, sr=16000) is not None]
    val_fake_features = [extract_features(audio_file, sr=16000) for audio_file in val_fake_audio_files if extract_features(audio_file, sr=16000) is not None]

    # Create labels
    real_labels = np.ones(len(real_features))
    fake_labels = np.zeros(len(fake_features))
    val_real_labels = np.ones(len(val_real_features))
    val_fake_labels = np.zeros(len(val_fake_features))

    # Combine the real and fake features and labels
    train_features = np.concatenate([real_features, fake_features])
    train_labels = np.concatenate([real_labels, fake_labels])
    val_features = np.concatenate([val_real_features, val_fake_features])
    val_labels = np.concatenate([val_real_labels, val_fake_labels])

    # Flatten and normalize features
    train_features_flat = np.array([feature.reshape(-1) for feature in train_features])
    val_features_flat = np.array([feature.reshape(-1) for feature in val_features])

    # Apply normalization to features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_flat)
    val_features_scaled = scaler.transform(val_features_flat)

    # Reshape features back to the original shape
    train_features_scaled = train_features_scaled.reshape((len(train_features), 20, max_length))
    val_features_scaled = val_features_scaled.reshape((len(val_features), 20, max_length))


    # Model Definition
    with tf.device('/device:GPU:0'):  # Specify GPU
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(20, max_length), return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    # Learning Rate Scheduling
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Model Compilation
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Print the model summary
    model.summary()

    # Model Training with Early Stopping
    with tf.device('/device:GPU:0'):  # Specify GPU
        print("Training started...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(train_features_scaled, train_labels, epochs=60, batch_size=32, validation_data=(val_features_scaled, val_labels), callbacks=[early_stopping])

    print("Training completed.")
    return model, history, scaler

# Paths to directories
real_audio_directory = "audio_dataset/training/real"
fake_audio_directory = "audio_dataset/training/fake"
val_real_audio_directory = "audio_dataset/validation/real"
val_fake_audio_directory = "audio_dataset/validation/fake"

# Train the model
trained_model, training_history, scaler = train_model(real_audio_directory, fake_audio_directory, val_real_audio_directory, val_fake_audio_directory)

# Save the trained model and scaler to files
trained_model.save("optimised_lstm_model")
joblib.dump(scaler, 'scaler.pkl')

# Print training history
print("History:")
print(training_history.history)

# Test the model on a separate test set
test_real_audio_directory = "audio_dataset/testing/real"
test_fake_audio_directory = "audio_dataset/testing/fake"

test_real_audio_files = load_audio_files_from_directory(test_real_audio_directory)
test_fake_audio_files = load_audio_files_from_directory(test_fake_audio_directory)

test_real_features = [extract_features(audio_file, sr=16000) for audio_file in test_real_audio_files if extract_features(audio_file, sr=16000) is not None]
test_fake_features = [extract_features(audio_file, sr=16000) for audio_file in test_fake_audio_files if extract_features(audio_file, sr=16000) is not None]

test_real_labels = np.ones(len(test_real_features))
test_fake_labels = np.zeros(len(test_fake_features))

test_features = np.concatenate([test_real_features, test_fake_features])
test_labels = np.concatenate([test_real_labels, test_fake_labels])

# Flatten and normalize test features
test_features_flat = np.array([feature.reshape(-1) for feature in test_features])
test_features_scaled = scaler.transform(test_features_flat)
test_features_scaled = test_features_scaled.reshape((len(test_features), 20, max_length))


# Evaluate the model on the test set
test_loss, test_accuracy = trained_model.evaluate(test_features_scaled, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
