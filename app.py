from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# 📁 Carpeta para guardar imágenes
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Crear carpeta si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 🔥 Cargar modelo entrenado
model = tf.keras.models.load_model("modelo/modelo.keras")

# 📌 Clases del modelo
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

# 📝 Descripciones de cada enfermedad
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

# 📐 Tamaño esperado por el modelo
IMG_SIZE = 224


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
            img_array = img_array / 255.0  # Normalización

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


if __name__ == "__main__":
    app.run(debug=True)