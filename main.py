import tensorflow as tf
import librosa
import numpy as np
from feedbacks import feedback

# Global constant
max_length = 204

# Feature Extraction Process
def extract_features(audio_file, sr=16000):
    audio, _ = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # Padding or truncation for consistent shape
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]

    return mfccs

# Load the trained model
model = tf.keras.models.load_model("Model/ann.h5")

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

# Example of making a prediction on a new audio file
new_audio_file = "enter the path of the audio clip to be analyzed" #enter the path of the audio file here
prediction = detect_fake_voice(new_audio_file, model)

# print(prediction)
print("\n",feedback(prediction))
print("Anyways, here is the prediction score: ", prediction)