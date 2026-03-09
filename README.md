# 🔍 DeepFake Image Detector

A deep learning web app that classifies images as **Real** or **AI-Generated (Fake)**.

🚀 **Live Demo:** [deepfake-detector-kasviii.streamlit.app](https://deepfake-detector-kasviii.streamlit.app)

## Model
- Architecture: MobileNetV2 (Transfer Learning)
- Dataset: 140,000+ images (Real vs Fake)
- Test Accuracy: 71.5%
- Framework: TensorFlow / Keras

## Results
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Fake  | 0.69      | 0.80   | 0.74     |
| Real  | 0.76      | 0.63   | 0.69     |

![Confusion Matrix](confusion_matrix.png)

## Limitations
Model was trained on GAN-based fakes. Performance on modern diffusion-based generators (Gemini, DALL-E, Midjourney) is lower as these post-date the training data.

## Stack
- Python, TensorFlow, Streamlit
- Deployed on Streamlit Community Cloud