import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Load model
model = tf.keras.models.load_model("OncoVision.h5")
class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# ---- Page Config ----
st.set_page_config(
    page_title="OncoVision",
    page_icon="🧠",
    layout="centered"
)

# ---- Helper to convert image to base64 ----
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---- Custom CSS ----
st.markdown("""
    <style>
        /* Black background */
        .stApp {
            background-color: black;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title */
        h1 {
            background: linear-gradient(90deg, #b68d2c, #e0c168, #f9e79f, #e0c168, #b68d2c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: bold;
        }
        .main-title {
        font-size: 60px;
        font-weight: bold;
        color: #f9e79f;  /* your gold gradient’s bright tone */
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .main-title span.icon {
        font-size: 55px;
        color: white; /* keep brain icon white, not gold */
    }
    .sub-title {
        font-size: 35px;
        color: white;
        text-align: center;
        margin-top: -5px;
    }
        /* File uploader */
        .stFileUploader label {
            background: linear-gradient(90deg, #b68d2c, #e0c168, #f9e79f, #e0c168, #b68d2c);
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 15px !important;
        }

        /* Uploaded image styling */
        .uploaded-img {
            display: block;
            margin: 20px auto;
            border: 4px solid #e0c168;
            border-radius: 12px;
            width: 70%; /* smaller than full width */
        }

        /* Prediction box */
        .result {
            padding: 20px;
            background: linear-gradient(90deg, #b68d2c, #e0c168, #f9e79f, #e0c168, #b68d2c);
            color: black;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
       html, body, [class*="block-container"], [data-testid="stAppViewContainer"] * {
            scrollbar-width: thin !important;
            scrollbar-color: #b68d2c transparent !important;
        }
        
        /* WebKit-based browsers */
        html::-webkit-scrollbar,
        body::-webkit-scrollbar,
        [class*="block-container"]::-webkit-scrollbar,
        [data-testid="stAppViewContainer"] *::-webkit-scrollbar {
            width: 12px !important;
            height: 12px !important;
        }
        
        html::-webkit-scrollbar-track,
        body::-webkit-scrollbar-track,
        [class*="block-container"]::-webkit-scrollbar-track,
        [data-testid="stAppViewContainer"] *::-webkit-scrollbar-track {
            background: transparent !important;
            border-radius: 10px !important;
        }
        
        html::-webkit-scrollbar-thumb,
        body::-webkit-scrollbar-thumb,
        [class*="block-container"]::-webkit-scrollbar-thumb,
        [data-testid="stAppViewContainer"] *::-webkit-scrollbar-thumb {
            background: linear-gradient(90deg, #b68d2c, #e0c168, #f9e79f, #e0c168, #b68d2c) !important;
            border-radius: 10px !important;
        }
        
        html::-webkit-scrollbar-thumb:hover,
        body::-webkit-scrollbar-thumb:hover,
        [class*="block-container"]::-webkit-scrollbar-thumb:hover,
        [data-testid="stAppViewContainer"] *::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(90deg, #f9e79f, #e0c168, #b68d2c, #e0c168, #f9e79f) !important;
        }

        button[kind="secondary"], button[kind="primary"], button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #b68d2c, #e0c168, #f9e79f, #e0c168, #b68d2c) !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6em 1.2em !important;
        cursor: pointer !important;
        }
    
        button[kind="secondary"]:hover, button[kind="primary"]:hover,
        button[data-testid="baseButton-secondary"]:hover, button[data-testid="baseButton-primary"]:hover {
            background: black !important;
            color: #f9e79f !important;
            border: 2px solid #e0c168 !important;
        }
        div[data-testid="stFileUploader"] {
        color: white !important;
        }
        [data-testid="stFileUploader"] svg {
            fill: #e0c168 !important;
            color: #e0c168 !important;
        }
        [data-testid="stFileUploader"] .uploadedFile span:nth-child(2) {
            color: #e0c168 !important;
            font-weight: bold;
        }
        [data-testid="stFileUploader"] small {
            color: #e0c168 !important;  /* gold */
            font-weight: bold !important;
        }

    </style>
    <div style="text-align: center;">
        <div class="main-title">
            <span class="icon">🧠</span> OncoVision
        </div>
        <div class="sub-title">
            Brain Tumor Classifier
        </div>
    </div>
    <br><br><br>
""", unsafe_allow_html=True)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image (smaller with gold border)
    img_display = Image.open(uploaded_file).convert("RGB")
    img_b64 = image_to_base64(img_display)
    st.markdown(
        f"<img src='data:image/png;base64,{img_b64}' class='uploaded-img'>",
        unsafe_allow_html=True
    )

    # Preprocess
    img = img_display.convert("L").resize((299, 299))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=(0, -1))

    # Prediction
    prediction = model.predict(x)
    predicted_class = class_names[np.argmax(prediction)]

    # Display Result
    st.markdown(f"<div class='result'>Prediction: {predicted_class}</div>", unsafe_allow_html=True)
