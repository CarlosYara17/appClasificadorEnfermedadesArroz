from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)

# ===============================
# CONFIGURACIÓN
# ===============================

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("modelo", exist_ok=True)

# Ruta donde se guardará el modelo
MODEL_PATH = "modelo/modelo_produccion.h5"

# 🔥 LINK DIRECTO DE DRIVE
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1OHBt_s3C-8AIoydj743M15ROWdMSRLpN"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Cargar modelo
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente")

# ===============================
# CLASES DEL MODELO
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

# ===============================
# DESCRIPCIONES
# ===============================

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

# ===============================
# RUTA PRINCIPAL
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    description = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_path = filepath

            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            predicted_class = class_names[class_index]

            prediction = f"{predicted_class} ({confidence*100:.2f}%)"
            description = descriptions.get(predicted_class)

    return render_template(
        "index.html",
        prediction=prediction,
        description=description,
        image_path=image_path
    )

# ===============================
# PRODUCCIÓN
# ===============================

if __name__ == "__main__":
    app.run()
