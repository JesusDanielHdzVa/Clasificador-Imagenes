import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Configuración de Flask ---
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Variable global para el modelo
modelo_global = None


def analizar_color(img):
    """Extrae media y desviación estándar de Hue, Saturation y Value, sirve para color"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v))


def analizar_textura_glcm(img):
    """Extrae Contraste, Homogeneidad y Entropía Global, sirve para texturas gruesas"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ubyte = img_as_ubyte(gray)

    entropia = shannon_entropy(gray)

    glcm = graycomatrix(
        gray_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contraste = graycoprops(glcm, "contrast")[0, 0]
    homogeneidad = graycoprops(glcm, "homogeneity")[0, 0]

    return contraste, homogeneidad, entropia


def analizar_textura_lbp(img):
    """Extrae Uniformidad y Entropía de LBP, esto sirve para texturas finas"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    uniformidad = np.sum(hist**2)
    entropia_lbp = -np.sum(hist * np.log2(hist + 1e-7))

    return uniformidad, entropia_lbp


def analizar_esquinas(img):
    """Calcula densidad de esquinas"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    esquinas = np.sum(dst > 0.01 * dst.max())
    densidad = esquinas / (img.shape[0] * img.shape[1])
    return densidad


def entrenar_modelo(ruta_csv):
    print(f" Cargando datos de: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    df.dropna(inplace=True)

    features = [
        "hue_mean",
        "hue_std",
        "sat_mean",
        "sat_std",
        "val_mean",
        "val_std",
        "contraste",
        "homogeneidad",
        "glcm_entropia",
        "lbp_uniformidad",
        "lbp_entropia",
        "esquinas",
    ]

    X = df[features]
    y = df["clase"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n ACCURACY DEL MODELO: {acc:.4f} ({(acc*100):.2f}%)")
    return clf


# Como no se mucho flask todo lo que sigue es de la estructura basica de flask segun Gemini
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "imagen" not in request.files:
        return redirect(url_for("home"))

    file = request.files["imagen"]
    if file.filename == "":
        return redirect(url_for("home"))

    if file and modelo_global is not None:
        try:
            # Leer la imagen
            filestream = file.read()
            nparr = np.frombuffer(filestream, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise Exception("No se pudo decodificar la imagen.")

            # Guardar imagen temporalmente
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            cv2.imwrite(filepath, img)

            # --- EXTRACCIÓN DE CARACTERÍSTICAS EN TIEMPO REAL ---
            # Ejecutamos las 4 funciones de análisis sobre la imagen nueva
            h_mean, h_std, s_mean, s_std, v_mean, v_std = analizar_color(img)
            cont, hom, entr_glcm = analizar_textura_glcm(img)
            lbp_uni, lbp_entr = analizar_textura_lbp(img)
            esquinas_val = analizar_esquinas(img)

            # Crear DataFrame con una sola fila (el orden de columnas debe ser IDENTICO al entrenamiento)
            features_list = [
                "hue_mean",
                "hue_std",
                "sat_mean",
                "sat_std",
                "val_mean",
                "val_std",
                "contraste",
                "homogeneidad",
                "glcm_entropia",
                "lbp_uniformidad",
                "lbp_entropia",
                "esquinas",
            ]

            valores = [
                [
                    h_mean,
                    h_std,
                    s_mean,
                    s_std,
                    v_mean,
                    v_std,
                    cont,
                    hom,
                    entr_glcm,
                    lbp_uni,
                    lbp_entr,
                    esquinas_val,
                ]
            ]

            df_prediccion = pd.DataFrame(valores, columns=features_list)

            # Predecir
            prediccion = modelo_global.predict(df_prediccion)[0]

            # Descripciones amigables
            mapa_resultados = {
                "Vegetacion": "🌿 Zona con vegetación detectada.",
                "Urbano": "🏙️ Zona urbana / Edificación detectada.",
                "Piel": "👤 Piel humana detectada.",
                "Madera": "🪵 Textura de madera detectada.",
                "Roca": "🪨 Superficie rocosa detectada.",
                "Tela": "👕 Textura de tela detectada.",
            }
            resultado_descriptivo = mapa_resultados.get(
                prediccion, f"Clase detectada: {prediccion}"
            )

            return render_template(
                "index.html",
                clase_predicha=resultado_descriptivo,
                imagen_url=filepath,
                # Enviamos algunos valores básicos para mostrar en pantalla si quieres
                intensidad_val=f"{s_mean:.2f}",
                contraste_val=f"{cont:.2f}",
            )

        except Exception as e:
            print(f"Error durante la predicción: {e}")
            return redirect(url_for("home"))

    return redirect(url_for("home"))


# --- 5. INICIO ---

try:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("--- 🤖 Iniciando sistema... ---")

    # IMPORTANTE: Asegúrate que este archivo exista
    archivo_datos = "resultadosV9.csv"

    if os.path.exists(archivo_datos):
        modelo_global = entrenar_modelo("resultadosV9.csv")
    else:
        print(
            f"⚠️ ADVERTENCIA: No se encontró '{archivo_datos}'. El modelo no funcionará."
        )

except Exception as e:
    print(f"Error fatal durante el inicio: {e}")

if __name__ == "__main__":
    if modelo_global:
        print("🚀 Servidor iniciado en http://127.0.0.1:5000")
        app.run(debug=True)
    else:
        print("❌ No se pudo cargar el modelo. Verifica el archivo CSV.")
