"""
Demostración: predecir la letra de una imagen externa (foto o dibujo propio).

Uso:
    python scripts/03_demo_imagen.py ruta/a/tu_imagen.png

A diferencia del dataset (que ya viene limpio), una foto propia trae:
  - fondo de cualquier color
  - la mano en cualquier parte del cuadro
  - iluminación desigual

Por eso aplicamos un preprocesamiento más agresivo: se detecta el contorno de la mano
(el objeto más oscuro o más claro según el caso), se recorta el cuadro alrededor de ella
y se centra. Es el mismo tipo de preparación que necesita una foto real.
"""
import os
import sys
import json
import numpy as np
import joblib
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO = os.path.join(BASE, "models", "m1_11letras_1capa.joblib")


def preprocesar(ruta, invertir=False, recortar=None):
    """Convierte una imagen externa al formato que espera el modelo (28x28, [0,1]).

    NO se invierte el contraste por defecto: el dataset tiene fondos de brillo muy
    variado, y se comprobó que invertir automáticamente estropea las imágenes cuyo
    fondo ya es oscuro (bajaba el acierto de 89% a 70%). El modelo se entrenó con las
    imágenes tal cual, así que la imagen se pasa tal cual. Solo si tu foto tiene el
    trazo claro sobre fondo oscuro y el modelo falla, prueba invertir=True.
    - recortar=None -> decisión automática: solo se recorta si la mano ocupa una
                       parte muy pequeña del cuadro (una foto tomada de lejos).
    """
    img = Image.open(ruta).convert("L")
    arr = np.array(img, dtype=np.float64)

    if invertir:
        arr = 255.0 - arr

    if recortar is None:
        # Solo se recorta si la mano ocupa una parte MUY pequeña del cuadro (una foto
        # tomada de lejos). Cualquier recorte cambia el encuadre respecto del dataset,
        # y el MLP es muy sensible a la posición del objeto (ver experimento de traslación).
        umbral = np.percentile(arr, 15) + 0.25 * (arr.max() - np.percentile(arr, 15))
        mascara = arr < umbral
        recortar = mascara.mean() < 0.15

    # 2. Recortamos alrededor del trazo para que la mano quede centrada
    if recortar:
        umbral = np.percentile(arr, 15) + 0.25 * (arr.max() - np.percentile(arr, 15))
        mascara = arr < umbral
        if mascara.sum() > 10:
            filas, cols = np.where(mascara)
            y0, y1 = filas.min(), filas.max()
            x0, x1 = cols.min(), cols.max()
            my = int((y1 - y0) * 0.15) + 1
            mx = int((x1 - x0) * 0.15) + 1
            y0, y1 = max(0, y0 - my), min(arr.shape[0], y1 + my)
            x0, x1 = max(0, x0 - mx), min(arr.shape[1], x1 + mx)
            arr = arr[y0:y1, x0:x1]

    # 3. Cuadrado, para no deformar la proporción de la mano
    h, w = arr.shape
    lado = max(h, w)
    lienzo = np.zeros((lado, lado), dtype=np.float64)
    lienzo[(lado - h) // 2:(lado - h) // 2 + h, (lado - w) // 2:(lado - w) // 2 + w] = arr
    arr = lienzo

    # 4. Redimensionar a 28x28 y normalizar a [0,1]
    img28 = Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr28 = np.array(img28, dtype=np.float64) / 255.0
    return arr28


def predecir(ruta):
    paquete = joblib.load(MODELO)
    modelo, nombres = paquete["model"], paquete["nombres"]

    arr28 = preprocesar(ruta)
    x = arr28.reshape(1, -1)
    prob = modelo.predict_proba(x)[0]
    orden = np.argsort(prob)[::-1]

    print(f"\nImagen analizada : {ruta}")
    print(f"Modelo           : {os.path.basename(MODELO)} ({len(nombres)} clases)")
    print("-" * 58)
    for k in range(min(3, len(nombres))):
        i = orden[k]
        barra = "#" * int(prob[i] * 40)
        etiqueta = "  <-- PREDICCION" if k == 0 else ""
        print(f"  {nombres[i]:>2} : {prob[i]*100:5.1f}%  {barra}{etiqueta}")
    print("-" * 58)
    letra = nombres[orden[0]]
    conf = prob[orden[0]] * 100
    if conf >= 70:
        print(f"El modelo cree que la letra es: {letra}  (confianza alta: {conf:.1f}%)")
    elif conf >= 40:
        print(f"El modelo cree que la letra es: {letra}  (confianza media: {conf:.1f}%)")
    else:
        print(f"Prediccion poco fiable: {letra} ({conf:.1f}%). "
              f"Recomendacion: abstenerse y pedir otra imagen.")
    # vista de lo que realmente ve el modelo
    print("\nLo que ve el modelo (8x8 caracteres):")
    for fila in arr28[::4, ::4]:
        print("   " + "".join(" .:-=+*#@"[min(8, int(v * 8.99))] for v in fila))
    return letra, float(conf)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Ejemplo: python scripts/03_demo_imagen.py data/amer_sign3.png")
        sys.exit(1)
    predecir(sys.argv[1])
