import streamlit as st
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

st.set_page_config(page_title="DeepFake Detector", page_icon="🔍", layout="centered")
st.title("🔍 DeepFake Image Detector")
st.markdown("Upload an image to check if it's **AI-generated (Fake)** or **Real**.")
st.divider()

@st.cache_resource
def load_model():
    model_path = "deepfake_model.h5"
    if not os.path.exists(model_path):
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1yrivvkZ4IOkv-dM2tNwi5Zv1naFKppHE",
            model_path, quiet=False
        )
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(model_path)
    return model

with st.spinner("Loading model..."):
    model = load_model()

st.success("Model ready!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    with st.spinner("Analysing..."):
        prediction = model.predict(img_array)[0][0]
    st.divider()
    if prediction > 0.5:
        st.error(f"🚨 REAL image — Confidence: {prediction*100:.1f}%")
    else:
        st.success(f"⚠️ FAKE / AI-Generated — Confidence: {(1-prediction)*100:.1f}%")
    st.caption("Model: MobileNetV2 · Trained on 140k images · Accuracy: 71.5%")
