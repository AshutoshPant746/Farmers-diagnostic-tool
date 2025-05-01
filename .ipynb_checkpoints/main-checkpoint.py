import streamlit as st
import tensorflow as tf
import numpy as np
import os

# ================== CLASS LABELS ==================
plant_class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                     'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
                     'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
                     'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight',
                     'Grape___healthy', 'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
                     'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                     'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
                     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

animal_class_names = ['Healthy Cow', 'Lumpy Skin Disease']
soil_class_names = ['Black Soil', 'Red Soil', 'Clay Soil', 'Yellow Soil']
soil_crop_map = {
    'Black Soil': 'Cotton, Soybean, Sorghum',
    'Red Soil': 'Groundnut, Millets, Pulses',
    'Clay Soil': 'Rice, Wheat, Sugarcane',
    'Yellow Soil': 'Pulses, Oilseeds'
}

# ================== MODEL LOADER ==================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.stop()
    return tf.keras.models.load_model(path, compile=False)

# ================== PREDICTION FUNCTIONS ==================
def predict_plant_disease(image):
    model = load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    input_arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)
    prediction = model.predict(input_arr)
    return np.argmax(prediction), float(np.max(prediction))

def predict_animal_disease(image):
    model = load_model('animal_skin_transfer4.keras')
    image = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    input_arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image) / 255.0, axis=0)
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

def predict_soil_type(image):
    model = load_model('final_soil_classification_model.h5')
    image = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    input_arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ================== UI CONFIG ==================
st.set_page_config(page_title="Farmer Diagnostic Tool", layout="wide")

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>✫ Farmer's Diagnostic Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Empowering farmers with AI to detect diseases and improve crop yield</p>", unsafe_allow_html=True)
st.markdown("---")

# ================== NAVIGATION ==================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🧪 Diagnose", "ℹ️ About"])

# ================== HOME TAB ==================
with tab1:
    st.subheader("🌿 Introduction")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    ### 🌾 Revolutionizing Farming with AI

    Agriculture is evolving. Farmers today need faster, smarter, and more accessible tools to tackle challenges in crop production, animal health, and land use.

    The **Farmer Diagnostic Tool** is an AI-powered assistant that uses deep learning models to:
    - 🧠 **Diagnose plant diseases** from leaf images.
    - 🐄 **Detect animal skin infections** using advanced computer vision.
    - 🌱 **Classify soil types** to guide better crop selection.

    Our goal is to bridge the technological gap in farming by offering real-time, offline-capable AI diagnostics directly at the hands of farmers.
    """)

    st.subheader("🚜 Why This Tool Matters")
    st.markdown("""
    - 🌾 **Boost Crop Yields:** Early detection of plant diseases prevents losses.
    - 🐄 **Enhance Animal Health:** Quick identification of livestock infections improves treatment timelines.
    - 🌱 **Optimize Soil Use:** Understanding soil type ensures crops are matched for better yield and soil sustainability.
    - 📱 **Simple & Accessible:** Designed to work on mobile devices even in low-connectivity regions.
    - 🧪 **Scientific Accuracy:** Uses high-accuracy CNN models trained on thousands of labeled images.
    """)

# ================== DIAGNOSTICS TAB ==================
with tab2:
    st.header("🔍 Farmer's Diagnostic Tool")

    col1, col2 = st.columns([1, 2])
    with col1:
        task = st.selectbox("🧬 Choose Diagnostic Task:", [
            "🌿 Plant Disease Detection", "🐄 Animal Disease Detection", "🌱 Soil Classification"
        ])
        uploaded_file = st.file_uploader("📷 Upload an Image", type=['jpg', 'jpeg', 'png'])

    with col2:
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if uploaded_file and st.button("🚀 Run Diagnosis"):
        with st.spinner("Analyzing... Please wait ⏳"):
            if task == "🌿 Plant Disease Detection":
                index, confidence = predict_plant_disease(uploaded_file)
                label = plant_class_names[index]
                st.success(f"🌿 Disease Detected: **{label}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(int(confidence * 100))

            elif task == "🐄 Animal Disease Detection":
                index = predict_animal_disease(uploaded_file)
                label = animal_class_names[index]
                st.warning(f"🐄 Detected Condition: **{label}**")

            elif task == "🌱 Soil Classification":
                index = predict_soil_type(uploaded_file)
                soil_type = soil_class_names[index]
                recommended = soil_crop_map[soil_type]
                st.success(f"🧪 Soil Type: **{soil_type}**")
                st.info(f"🌾 Recommended Crops: **{recommended}**")

# ================== ABOUT TAB ==================
with tab3:
    st.subheader("📖 Project Overview")
    st.markdown("""
    ### Home

    Welcome to the **Farmer Diagnostic Tool**, your smart assistant for diagnosing plant and animal health and understanding soil composition.  
    This app uses advanced AI-powered image classification models to assist you in three major agricultural areas:

    - 🌿 **Plant Disease Detection**: Upload plant leaf images to identify possible diseases.
    - 🐄 **Animal Skin Disease Detection**: Upload livestock skin images to detect common skin diseases.
    - 🌱 **Soil Classification**: Upload soil images to determine soil type and get suitable crop suggestions.

    This tool is built for farmers, agriculturists, and gardening enthusiasts to make informed decisions based on instant analysis.

    ---

    ### About

    #### System Overview

    Agriculture and gardening are fundamental for food production and personal cultivation, yet both face challenges such as disease detection and soil analysis.  
    The **Farmer Diagnostic Tool** is a web-based application that assists users through advanced image classification techniques.  
    It uses CNNs with transfer learning to provide reliable predictions and recommendations for plant health, animal conditions, and soil suitability.

    #### Introduction

    Traditional diagnostics in agriculture require time and expertise, which may not always be available—especially in rural areas.  
    This tool provides a simple web interface for uploading images and getting real-time analysis for:
    
    - Plant disease diagnosis.
    - Animal skin condition detection.
    - Soil type classification with crop suggestions.

    #### Problem Statement

    Users face challenges like:

    - ❗ **Delayed Disease Detection**: Slows intervention for crop and livestock diseases.
    - ❗ **Lack of Expert Access**: Especially in remote farming communities.
    - ❗ **Uncertain Soil Suitability**: Hinders optimal crop selection.
    - ❗ **Limited Knowledge of Animal Skin Diseases**: Increases risk of spread and delayed care.

    #### Objective

    - Build a Streamlit-based web app to upload and analyze images in real-time.
    - Apply CNN and MobileNetV2 models to detect:
        - Plant diseases.
        - Animal skin diseases.
        - Soil types and suggest crops.
    - Deliver recommendations to aid quick and informed decisions.

    #### Algorithms Used

    **1. Convolutional Neural Networks (CNNs):**
    - Automatically extract features from plant leaves, animal skin, and soil textures.

    **2. Transfer Learning (MobileNetV2):**
    - Lightweight and efficient.
    - Suited for mobile/web deployment.
    - Performs well with smaller datasets.

    #### Literature Survey

    - **Plant Diseases**: CNNs effectively detect visual symptoms on leaves.
    - **Animal Diseases**: MobileNet/DenseNet outperform traditional ML models.
    - **Soil**: RGB-based classification using smartphone images shows promising results.

    #### Methodology

    1. **Data Collection & Preprocessing**: Normalize, resize, and augment datasets.
    2. **Model Training**: Train CNNs and fine-tune MobileNetV2 for all three domains.
    3. **Web App Development**: Streamlit app to load models and run predictions.
    4. **Evaluation**: Use accuracy, F1-score, and tuning to improve performance.

    #### Implementation

    - **Plant Disease**: CNN on 87K leaf images (Kaggle); dropout + Adam optimizer.
    - **Animal Skin**: Custom CNN + MobileNetV2; focused on lumpy skin disease.
    - **Soil**: Texture-based classification + crop suggestions based on output.

    #### Sample Use Cases

    - A farmer detects early powdery mildew from a leaf photo and gets treatment tips.
    - A livestock owner spots lumpy skin disease and seeks early vet advice.
    - A gardener learns their soil is red and suitable for tomatoes and legumes.

    #### Future Enhancements

    - 🌐 Multilingual support for wider adoption.
    - 📱 Mobile app for offline usage.
    - 🌾 Add more disease classes and soil types.

    ---
    For further documentation, refer to the README or contact the development team.
    """)

