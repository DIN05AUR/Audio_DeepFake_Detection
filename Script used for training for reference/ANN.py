import tensorflow as tf
import librosa
import numpy as np
import os


max_length = 204  # still searching for the best fit value for this; default value = 204

#Feature Extraction Process
def extract_features(audio_file, sr=16000):
    # inputting the audio file
    audio, _ = librosa.load(audio_file, sr=sr)

    #actual extraction of MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    #padding aur truncation for consistent shape
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]

    return mfccs

# load the training data
def load_audio_files_from_directory(directory):
    audio_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.wav')]
    return audio_files

# training process starts here...........
def train_model():
    # loading the training data
    real_audio_directory = "audio_dataset/training/real"  # Change this to your real audio directory
    fake_audio_directory = "audio_dataset/training/fake"  # Change this to your fake audio directory

    real_audio_files = load_audio_files_from_directory(real_audio_directory)
    fake_audio_files = load_audio_files_from_directory(fake_audio_directory)

    # extraction of features from the training data
    real_features = np.array([extract_features(audio_file, sr=16000) for audio_file in real_audio_files])
    fake_features = np.array([extract_features(audio_file, sr=16000) for audio_file in fake_audio_files])

    # Create labels
    real_labels = np.ones(len(real_features))
    fake_labels = np.zeros(len(fake_features))

    # Combine the real and fake features and labels
    features = np.concatenate([real_features, fake_features])
    labels = np.concatenate([real_labels, fake_labels])

    # Create a neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(20, max_length)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Now lets compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model for some epochs
    model.fit(features, labels, epochs=100)

    return model

# Function to detect a fake voice
def detect_fake_voice(audio_file, model):

    # Extract features from the audio
    mfccs = extract_features(audio_file, sr=16000)

    # Ensure consistent shape (padding or truncating if necessary)
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]

    # Predict the label of the audio
    prediction = model.predict(np.expand_dims(mfccs, axis=0))

    # Return the prediction
    return prediction

# Train the model
model = train_model()

# Save the trained model to a file
model.save("ann_new_model.h5")