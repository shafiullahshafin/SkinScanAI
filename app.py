import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.header('Skin Lesion Classification System')
disease_names = [
    'Actinic Keratoses and Intraepithelial Carcinoma (AKIEC)',
    'Basal Cell Carcinoma (BCC)',
    'Benign Keratosis-like Lesions (BKL)',
    'Dermatofibroma (DF)',
    'Melanoma (MEL)',
    'Melanocytic Nevi (NV)',
    'Vascular Lesions (VASC)']

model = load_model('DN_Model.h5')

from tensorflow.keras.applications.densenet import preprocess_input

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_array = preprocess_input(input_image_array) 
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f'**Predicted Class:** {disease_names[np.argmax(result)]}\n\n**Prediction Score:** {np.max(result) * 100:.2f}%'
    return outcome

upload_dir = 'Uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    file_path = os.path.join('Uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width = 200)
    outcome = classify_images(file_path)
    st.markdown(outcome)
