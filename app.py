import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.metrics import MeanAbsoluteError


# Load model and normalization parameters
model = load_model("gdp_model.h5", custom_objects={'mae': MeanAbsoluteError()})
translation = np.load("translation.npy", allow_pickle=True)
scale = np.load("scale.npy", allow_pickle=True)

# Define a poverty threshold (adjust this based on your dataset's range)
poverty_threshold = 25000

# App configuration
st.set_page_config(page_title="GDP & Poverty Predictor", layout="centered")
st.title("🌍 GDP & Poverty Prediction from Satellite Nightlight Image")

# Upload section
uploaded_file = st.file_uploader("Upload a grayscale satellite nightlight image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Layout for side-by-side images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, caption="Original Nightlight Image", use_container_width=True)

    # Histogram Equalization
    equalized_image = cv2.equalizeHist(image)
    with col2:
        st.subheader("✨ Equalized Image")
        st.image(equalized_image, caption="Equalized Image", use_container_width=True)

    # Preprocess for model input
    img_input = cv2.resize(equalized_image, (256, 256))
    img_input = img_input.reshape((1, 256, 256, 1))

    # Predict normalized GDP and rescale
    prediction_norm = model.predict(img_input)[0][0]
    gdp_value = prediction_norm * scale + translation

    # Show prediction
    st.success(f"📊 Predicted GDP: ${gdp_value:,.2f}")

    # Classify wealth status
    if gdp_value < poverty_threshold:
        st.error("🧾 Predicted as a Poverty-Prone Region")
    else:
        st.success("💰 Predicted as a Wealthy/Developed Region")
