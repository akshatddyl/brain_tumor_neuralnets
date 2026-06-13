import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠")
st.title("🧠 CNN Brain Tumor Detection AI")
st.write("Upload a brain MRI scan and let the deep learning model detect anomalies.")

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_cnn_model():
    # Make sure you place the brain_tumor_model.h5 (downloaded from Colab) in the same folder as this script.
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model
    except Exception as e:
        st.error("Model not found! Please make sure 'brain_tumor_model.h5' is in the same directory.")
        return None

model = load_cnn_model()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a Brain MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded MRI Scan")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("AI Analysis Result")
        with st.spinner("Analyzing scan..."):
            # Preprocess the image to match the Colab training data
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.resize(img_array, (150, 150)) # Resize
            img_array = img_array / 255.0                 # Normalize
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            
            # Predict
            prediction = model.predict(img_array)[0][0]
            
            # Display logic
            st.markdown("---")
            if prediction > 0.5:
                st.error("### 🚨 High Probability of Brain Tumor")
                st.write(f"**Confidence:** {prediction*100:.2f}%")
                st.info("Recommendation: Consult a medical professional immediately.")
            else:
                st.success("### ✅ No Tumor Detected")
                st.write(f"**Confidence:** {(1-prediction)*100:.2f}%")
                st.info("The scan appears to be healthy.")
