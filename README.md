
# Audio_DeepFake_Detection 🔉🔎

This repository presents an AI model built using deep learning techniques, developed to distinguish between genuine audio clips and synthetically generated audioclips (audio deepfakes). The model is trained to identify subtle characteristics that differentiate real and AI-manipulated audio, aiding in the fight against misinformation and deception.


## Research Paper 📑

**TITLE: SYNTHETIC SPEECH DETECTION USING DEEP LEARNING**

**ABSTRACT:**
The evolution from harmless text-to-speech (TTS) systems to deceptive audio deepfakes, the AI generated audio
became a threat capable of impersonation and spreading misinformation. Deep learning (DL) techniques has
exacerbated this danger, crafting ever-more convincing fabrications necessitating robust detection methods.
While existing research has achieved impressive results, their real-world generalizability remains limited due to
single-dataset training and assessment, impractical for representing the diverse landscape of modern audio
deepfakes. This paper tackles the same problem by strategically combining diverse audio data and contructing
a dynamic training superset. This incorporates enriching the widely used Fake or Real (FoR) dataset with
diverse audio data, ensuring the model adaptability across genders, languages and accents along with several
TTS systems. This departure from prior research highlights our focus on real-world generalizability. We
experiment with DL methods like artifical neural networks (ANNs) and long short term memory (LSTMs). Our
model, leveraging MFCC (Mel Frequency Cepstrum Coefficient) features and LSTM networks, achieves
benchmark accuracy of 85.5% and exhibits promising real-world generalizability, demonstrating its potential to
address this critical challenge.

**JOURNAL:** International Journal of Advance and Innovative Research, Volume 11, Issue 1 (XIV): January - March 2024

**REFERENCE LINK:** https://iaraedu.com/pdf/ijair-volume-11-issue-1-xiv-january-march-2024.pdf [Page 230]
## Dataset Details 📊

The primary dataset used to train this model was the Fake-or-Real (FoR) dataset, which was augmented and enriched with additional well-known audio datasets such as Mozilla Common Voice and various Text-to-Speech (TTS) system datasets.

Due to the original dataset's size of 25GB, the dataset folder includes a small representative sample of the complete dataset. It contains three folders: testing, training, and validation, each consisting of both fake and real samples.

![Dataset List](https://github.com/DIN05AUR/Audio_DeepFake_Detection/blob/master/Some%20Images%20(ignore)/Dataset%20Details.jpg)

Fake-or-Real(FoR) Dataset Link:
https://bil.eecs.yorku.ca/datasets/
## Models and their Metrics 📶

To achieve higher accuracy in determining whether an audio clip is real or AI-generated, we trained multiple models. Among them, three models stood out in terms of performance:

- **ANN Model:** Initially, we trained an Artificial Neural Network (ANN) model.

- **Optimized ANN Model:** We then optimized the ANN model for better performance and more accurate predictions.

- **LSTM Model:** After fully optimizing the ANN model to it's max potential and still not meeting our expectations, we transitioned to a Long Short-Term Memory (LSTM) neural network architecture.

The trained (ready-to-use) models are available in the Model/ folder. Here are the metrics of the trained models.
![Model Metrics](https://github.com/DIN05AUR/Audio_DeepFake_Detection/blob/master/Some%20Images%20(ignore)/Model%20Metrics.jpg)
## Installation 💻

**Steps to Use the Model:**

**1] Install Necessary Libraries:** Ensure all required libraries are installed, including TensorFlow (TF), Librosa, Keras, NumPy, etc.

**2] Update main.py Script:**

- At line 45, provide the path of the audio file for prediction.
- At line 24, enter the path of the model you're going to use for making predictions (default path is already set).

**3] Make Predictions:** Run the script to generate predictions.

    
## Screenshots 📸

![Output Screenshot](https://github.com/DIN05AUR/Audio_DeepFake_Detection/blob/master/Some%20Images%20(ignore)/Output_Screenshot.jpg)
## Author 👨🏻‍💻

- [@DIN05AUR](https://github.com/DIN05AUR)

Contact at: saurabhg.k.221@gmail.com

