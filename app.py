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
    "Bacterial Leaf Blight": "Provocado por la bacteria Xanthomonas oryzae pv. oryzae, este patógeno genera la decoloración progresiva del borde de las hojas, que inicia como rayas acuosas amarillentas y se extiende hacia el centro.",
    "Brown Spot": "Causada por el hongo Bipolaris oryzae, se aprecia en forma de manchas redondeadas de color pardo oscuro con centros más claros. Es común en suelos pobres en nutrientes, especialmente con deficiencia de nitrógeno, y puede afectar tanto las hojas como las vainas, reduciendo la fotosíntesis y la calidad del grano.",
    "Healthy Rice Leaf": "Hoja sana sin signos visibles.",
    "Leaf Blast": "Causado por el hongo Magnaporthe oryzae (sin. Pyricularia oryzae), es una de las enfermedades más destructivas del arroz. Se caracteriza por la aparición de lesiones scon centros grises y bordes marrones.",
    "Leaf scald": "Originada por el hongo Microdochium oryzae, esta enfermedad produce lesiones irregulares de color pardo rojizo con bordes amarillos que dan la apariencia de hojas “escaldadas”.",
    "Narrow Brown Leaf Spot": "Provocada por el hongo Cercospora oryzae, se distingue por generar manchas alargadas y angostas de color pardo oscuro, generalmente paralelas a las nervaduras de la hoja.",
    "Sheath Blight": "Producido por el hongo Rhizoctonia solani, afecta principalmente las vainas de las hojas cercanas al agua. Se caracteriza por manchas ovaladas o irregulares de color gris pardo con bordes oscuros, que se expanden rápidamente bajo condiciones cálidas y húmedas.",
    "Rice Hispa": "Causada por el insecto Dicladispa armigera, conocido como “hispa del arroz”, esta plaga raspa el tejido foliar, dejando líneas paralelas blanquecinas en la superficie."
}

# ===============================
# INTERFAZ
# ===============================

st.title("Clasificador de Enfermedades del Arroz")

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

