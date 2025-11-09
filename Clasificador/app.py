import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Nose como funciona flask esto es chat gpt generado
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# analiza el color
def color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    return np.mean(s)


# Analiza la textura
def textura(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ubyte = img_as_ubyte(gray)
    glcm = graycomatrix(
        gray_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    return graycoprops(glcm, "contrast")[0, 0]


def entrenar_modelo(ruta_csv):
    df = pd.read_csv(ruta_csv)

    X = df[["color", "contraste"]]
    y = df["clase"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=4
    )

    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
    return clf


@app.route("/")
def home():
    # Mostrar la página principal
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Logica de la predicción
    if "imagen" not in request.files:
        return redirect(url_for("home"))

    file = request.files["imagen"]

    if file.filename == "":
        return redirect(url_for("home"))

    if file and modelo_global is not None:
        filestream = file.read()
        nparr = np.frombuffer(filestream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("No se pudo decodificar la imagen.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        cv2.imwrite(filepath, img)

        # Analizar los colores
        intensidad = color(img)
        contraste = textura(img)

        # Prediccion de la imagen
        datos_para_predecir = pd.DataFrame(
            [[intensidad, contraste]], columns=["color", "contraste"]
        )
        prediccion = modelo_global.predict(datos_para_predecir)[0]

        # Diccionario para mapear las clases a frases descriptivas
        mapa_resultados = {
            "Vegetacion": "En este lugar hay presencia de vegetación.",
            "Urbano": "El modelo detectó una zona urbana.",
            "Piel": "El modelo detectó piel.",
            "Madera": "El modelo detectó madera.",
            "Roca": "El modelo detectó una roca.",
        }

        resultado_descriptivo = mapa_resultados.get(
            prediccion, f"Clase detectada: {prediccion}"
        )

        # ------------------------------------------

        # Mostrar el resultado
        return render_template(
            "index.html",
            clase_predicha=resultado_descriptivo,
            imagen_url=filepath,
            intensidad_val=f"{intensidad:.2f}",
            contraste_val=f"{contraste:.2f}",
        )

    return redirect(url_for("home"))


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    modelo_global = entrenar_modelo("resultadosV4.csv")

    if modelo_global:
        print("Iniciando servidor Flask en http://127.0.0.1:5000")
        app.run(debug=True)
    else:
        print("No se pudo cargar el modelo. El servidor no se iniciará.")
