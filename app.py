import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown

# ===============================
# CONFIGURACIÓN DE PÁGINA
# ===============================
st.set_page_config(page_title="Detector de Enfermedades en Arroz", layout="centered")

# ===============================
# ESTILOS CSS (simula tu HTML)
# ===============================
st.markdown(
"""
<style>
body {
    background: linear-gradient(135deg, #2e7d32, #81c784);
    font-family: 'Poppins', sans-serif;
}
.header {
    background-color: rgba(0,0,0,0.2);
    padding: 20px;
    text-align: center;
    color: white;
    border-radius: 10px;
}
.container {
    background: white;
    width: 90%;
    max-width: 900px;
    margin: 40px auto;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
    text-align: center;
}
input[type="file"] {
    margin: 20px 0;
}
.result-box {
    margin-top: 30px;
    padding: 25px;
    background-color: #f1f8e9;
    border-left: 6px solid #2e7d32;
    border-radius: 10px;
    text-align: left;
}
.result-box h2 {
    margin-top: 0;
    color: #2e7d32;
}
.confidence {
    font-weight: bold;
    color: #1b5e20;
}
footer {
    text-align: center;
    padding: 15px;
    color: white;
    font-size: 14px;
    margin-top: 20px;
}
</style>
""",
unsafe_allow_html=True
)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="header"><h1>Sistema Inteligente de Diagnóstico en Arroz</h1><p>Clasificación automática mediante Deep Learning</p></div>', unsafe_allow_html=True)

# ===============================
# DIRECTORIOS Y MODELO
# ===============================
os.makedirs("modelo", exist_ok=True)
MODEL_PATH = "modelo/modelo_produccion.h5"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1OHBt_s3C-8AIoydj743M15ROWdMSRLpN"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    st.write("Descargando modelo desde Google Drive...")
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
# CLASES Y DESCRIPCIONES
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

descriptions = {
    "Bacterial Leaf Blight": "Enfermedad bacteriana que provoca marchitez y secado progresivo de las hojas.",
    "Brown Spot": "Causada por hongos, genera manchas marrones circulares en la hoja.",
    "Healthy Rice Leaf": "Hoja sana sin signos visibles de enfermedad.",
    "Leaf Blast": "Enfermedad fúngica que produce lesiones en forma de diamante.",
    "Leaf scald": "Provoca decoloración y apariencia de hoja quemada.",
    "Narrow Brown Leaf Spot": "Genera manchas alargadas y estrechas de color marrón.",
    "Sheath Blight": "Enfermedad fúngica que afecta la vaina y reduce rendimiento.",
    "Rice Hispa": "Plaga que daña la superficie de la hoja dejando líneas blancas."
}

IMG_SIZE = 224

# ===============================
# CONTENEDOR PRINCIPAL
# ===============================
st.markdown('<div class="container">', unsafe_allow_html=True)
st.write("### Sube una imagen de la hoja de arroz")

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen subida", width=300)

    # Preprocesar imagen
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = class_names[class_index]

    # Mostrar resultado
    st.markdown(
        f"""
        <div class="result-box">
            <h2>Resultado del Análisis</h2>
            <p><strong>Diagnóstico:</strong> {predicted_class} ({confidence*100:.2f}%)</p>
            <p><strong>Descripción:</strong> {descriptions[predicted_class]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown('<footer>Proyecto de Clasificación de Enfermedades en Arroz | Deep Learning</footer>', unsafe_allow_html=True)
