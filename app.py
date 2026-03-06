import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import gdown
from PIL import Image

# ===============================
# CONFIGURACIÓN
# ===============================

MODEL_PATH = "modelo/modelo_produccion.h5"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1OHBt_s3C-8AIoydj743M15ROWdMSRLpN"

os.makedirs("modelo", exist_ok=True)

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    st.write("Descargando modelo...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# ===============================
# CARGAR MODELO
# ===============================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ===============================
# CLASES
# ===============================

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Narrow Brown Leaf Spot",
    "Sheath Blight",
    "Rice Hispa"
]

IMG_SIZE = 224

descriptions = {
    "Bacterial Leaf Blight": "Enfermedad bacteriana que provoca marchitez y secado progresivo de las hojas.",
    "Brown Spot": "Causada por hongos, genera manchas marrones circulares.",
    "Healthy Rice Leaf": "Hoja sana sin signos visibles.",
    "Leaf Blast": "Lesiones en forma de diamante.",
    "Leaf scald": "Apariencia de hoja quemada.",
    "Narrow Brown Leaf Spot": "Manchas marrones alargadas.",
    "Sheath Blight": "Enfermedad fúngica que afecta la vaina.",
    "Rice Hispa": "Plaga que deja líneas blancas."
}

# ===============================
# INTERFAZ
# ===============================

st.title("🌾 Clasificador de Enfermedades del Arroz")

uploaded_file = st.file_uploader("Sube una imagen de la hoja", type=["jpg","png","jpeg"])

if uploaded_file:

    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen subida", use_column_width=True)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    predicted_class = class_names[class_index]

    st.success(f"Predicción: {predicted_class}")
    st.write(f"Confianza: {confidence*100:.2f}%")
    st.info(descriptions[predicted_class])
