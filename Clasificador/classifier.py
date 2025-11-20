import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # ¡Mucho mejor que DecisionTree!
from sklearn.metrics import accuracy_score, classification_report


def entrenar_clasificador_avanzado(ruta_csv):
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


if __name__ == "__main__":
    modelo = entrenar_clasificador_avanzado("resultadosV9.csv")
