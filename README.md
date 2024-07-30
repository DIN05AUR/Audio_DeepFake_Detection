This repository presents an AI model built using deep learning techniques model developed to distinguish between genuine audio clips and synthetically generated audio (audio deepfakes). The model is trained to identify subtle characteristics that differentiate real and AI-manipulated audio, aiding in the fight against misinformation and deception.

The primary dataset used to train this model was the Fake-or-Real (FoR) dataset, which was augmented and enriched with additional well-known audio datasets such as Mozilla Common Voice and various Text-to-Speech (TTS) system datasets.

Due to the original dataset's size of 25GB, the dataset folder includes a representative sample of the complete dataset. It contains three folders: testing, training, and validation. Each folder consists of both fake and real samples.

Steps to Use the Model:
1. Install Necessary Libraries: Ensure all required libraries are installed, including TensorFlow (TF), Librosa, Keras, NumPy, etc.
2. Update main.py Script:
  1) At line 45, provide the path of the audio file for prediction.
  2) At line 24, enter the path of the model (default path is already set).
3. Make Predictions: Run the script to generate predictions.
