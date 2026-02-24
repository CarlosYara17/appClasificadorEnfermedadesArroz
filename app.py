from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)

# ==============================
# 📁 CONFIGURACIÓN DE CARPETAS
# ==============================

UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "modelo"
MODEL_PATH = os.path.join(MODEL_FOLDER, "inception_best.keras")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Crear carpetas si no existen
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# ==============================
# 🔗 ENLACE GOOGLE DRIVE
# ==============================

DRIVE_LINK = "https://drive.google.com/uc?id=16tJjVLj3ZkJ3uH2WsFUfKBvuFPUe0gGn"

# Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)

# ==============================
# 🧠 CARGAR MODELO
# ==============================

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente.")

# ==============================
# 📌 CLASES
# ==============================

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

# ==============================
# 📝 DESCRIPCIONES
# ==============================

descriptions = {
    "Bacterial Leaf Blight": "Enfermedad bacteriana que provoca marchitez y manchas amarillentas en los bordes de la hoja. Puede reducir significativamente el rendimiento del cultivo.",
    
    "Brown Spot": "Enfermedad fúngica caracterizada por manchas marrones circulares en las hojas. Es común en suelos con deficiencia nutricional.",
    
    "Healthy Rice Leaf": "La hoja se encuentra en estado saludable, sin presencia de lesiones, manchas o signos de infección.",
    
    "Leaf Blast": "Enfermedad causada por el hongo Magnaporthe oryzae que genera lesiones en forma de diamante en las hojas.",
    
    "Leaf scald": "Produce manchas alargadas y blanquecinas que pueden expandirse y secar grandes áreas de la hoja.",
    
    "Narrow Brown Leaf Spot": "Se caracteriza por manchas delgadas y marrones distribuidas en la superficie de la hoja.",
    
    "Sheath Blight": "Infección fúngica que afecta la vaina de la planta, generando lesiones ovaladas y disminuyendo la productividad.",
    
    "Rice Hispa": "Plaga causada por insectos que raspan la superficie de la hoja, provocando líneas blancas y debilitamiento de la planta."
}

# Tamaño esperado por el modelo
IMG_SIZE = 224

# ==============================
# 🌐 RUTA PRINCIPAL
# ==============================

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

            # 🔄 Preprocesamiento
            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # 🔍 Predicción
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            predicted_class = class_names[class_index]
            prediction = f"{predicted_class} ({confidence*100:.2f}%)"
            description = descriptions[predicted_class]

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        description=description
    )

# ==============================
# 🚀 INICIAR APP
# ==============================

if __name__ == "__main__":
    app.run(debug=True)