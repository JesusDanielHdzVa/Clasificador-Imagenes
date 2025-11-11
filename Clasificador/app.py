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

# --- 1. Configuración de Flask ---
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Límite de 16MB

# --- ✨ 2. Definición de la Variable Global ---
# Esta línea debe estar aquí para que el script la conozca
modelo_global = None


# --- 3. Funciones de Análisis (Sin cambios) ---
def color(img):
    """analiza el color"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    return np.mean(s)


def textura(img):
    """Analiza la textura"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ubyte = img_as_ubyte(gray)
    glcm = graycomatrix(
        gray_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    return graycoprops(glcm, "contrast")[0, 0]


def entrenar_modelo(ruta_csv):
    """Función de entrenamiento"""
    try:
        df = pd.read_csv(ruta_csv)
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Error al leer o limpiar el CSV: {e}")
        return None

    X = df[["color", "contraste"]]
    y = df["clase"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4
    )

    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
    return clf


# --- 4. Rutas de la Aplicación (Sin cambios) ---
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

    # Ahora 'modelo_global' es accesible y está entrenado
    if file and modelo_global is not None:
        try:
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

            # --- Lógica de Traducción (como la teníamos) ---
            mapa_resultados = {
                "Vegetacion": "En este lugar hay presencia de vegetación.",
                "Urbano": "El modelo detectó una zona urbana.",
                # Añade aquí tus otras clases
            }
            resultado_descriptivo = mapa_resultados.get(
                prediccion, f"Clase detectada: {prediccion}"
            )

            # Mostrar el resultado en la misma página
            return render_template(
                "index.html",
                clase_predicha=resultado_descriptivo,
                imagen_url=filepath,
                intensidad_val=f"{intensidad:.2f}",
                contraste_val=f"{contraste:.2f}",
            )
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            return redirect(url_for("home"))

    return redirect(url_for("home"))


# --- ✨ 5. INICIO DE LA APP Y ENTRENAMIENTO ---
# Esta parte se ejecuta UNA VEZ cuando Render inicia el servidor.
# Está FUERA del bloque 'if __name__ ...'
try:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("--- 🤖 Entrenando modelo al iniciar el servidor... ---")
    modelo_global = entrenar_modelo("resultadosV4.csv")
    if modelo_global:
        print("--- ✅ Modelo entrenado y listo. ---")
    else:
        print("--- ❌ ERROR AL ENTRENAR EL MODELO. ---")
except Exception as e:
    print(f"Error fatal durante el inicio: {e}")


# El bloque 'if __name__ == "__main__"' solo se usa para pruebas locales
if __name__ == "__main__":
    if modelo_global:
        print("Iniciando servidor Flask de DESARROLLO en http://127.0.0.1:5000")
        app.run(debug=True)
    else:
        print("No se pudo cargar el modelo. El servidor local no se iniciará.")
