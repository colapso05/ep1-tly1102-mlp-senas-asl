"""
EDA del dataset Sign Language MNIST -> genera figuras en images/
Proyecto: EP1 TLY1102 - Clasificación de letras ASL con MLP
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
IMG = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# ---------------------------------------------------------------- carga
train = pd.read_csv(os.path.join(DATA, "sign_mnist_train.csv"))
test = pd.read_csv(os.path.join(DATA, "sign_mnist_test.csv"))

X_train_full = train.drop("label", axis=1).values.astype(np.float64)
y_train_full = train["label"].values
X_test_full = test.drop("label", axis=1).values.astype(np.float64)
y_test_full = test["label"].values

# El dataset omite 9 (J) y 25 (Z) porque requieren movimiento.
PRESENT = sorted(np.unique(y_train_full))
INT2LETRA = {i: chr(65 + i) for i in range(26)}
# remapeo a etiquetas contiguas 0..23 para la salida del MLP
REMAP = {orig: new for new, orig in enumerate(PRESENT)}

resumen = {
    "train_shape": list(train.shape),
    "test_shape": list(test.shape),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "n_features": int(X_train_full.shape[1]),
    "clases_originales": [int(c) for c in PRESENT],
    "clases_letras": [INT2LETRA[c] for c in PRESENT],
    "clases_ausentes": [int(c) for c in (9, 25)],
    "pixel_min": int(X_train_full.min()),
    "pixel_max": int(X_train_full.max()),
    "pixel_media": float(X_train_full.mean()),
    "pixel_std": float(X_train_full.std()),
    "nulos_train": int(train.isnull().sum().sum()),
    "nulos_test": int(test.isnull().sum().sum()),
}
print(json.dumps(resumen, indent=2, ensure_ascii=False))

# -------------------------------------------------- 1. distribución de clases
dist_tr = train["label"].value_counts().sort_index()
dist_te = test["label"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
etiquetas = [INT2LETRA[c] for c in dist_tr.index]
axes[0].bar(range(len(dist_tr)), dist_tr.values, color="#2c7bb6", edgecolor="black", linewidth=.4)
axes[0].axhline(dist_tr.mean(), color="#d7191c", ls="--", lw=1.4,
                label=f"Media = {dist_tr.mean():.0f} imgs/clase")
axes[0].set_title("Distribución de clases — entrenamiento (27.455 imgs)")
axes[0].set_xticks(range(len(dist_tr)))
axes[0].set_xticklabels(etiquetas, fontsize=8)
axes[0].set_xlabel("Letra (seña ASL)")
axes[0].set_ylabel("N° de imágenes")
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=.3)

axes[1].bar(range(len(dist_te)), dist_te.values, color="#fdae61", edgecolor="black", linewidth=.4)
axes[1].axhline(dist_te.mean(), color="#d7191c", ls="--", lw=1.4,
                label=f"Media = {dist_te.mean():.0f} imgs/clase")
axes[1].set_title("Distribución de clases — test (7.172 imgs)")
axes[1].set_xticks(range(len(dist_te)))
axes[1].set_xticklabels([INT2LETRA[c] for c in dist_te.index], fontsize=8)
axes[1].set_xlabel("Letra (seña ASL)")
axes[1].set_ylabel("N° de imágenes")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "01_distribucion_clases.png"), bbox_inches="tight")
plt.close()

# -------------------------------------------------- 2. ejemplos por clase
fig, axes = plt.subplots(4, 6, figsize=(11, 7.6))
for ax, orig in zip(axes.flatten(), PRESENT):
    idx = int(np.where(y_train_full == orig)[0][0])
    ax.imshow(X_train_full[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"{INT2LETRA[orig]}  (clase {orig})", fontsize=9)
    ax.axis("off")
fig.suptitle("Una imagen de entrenamiento por cada una de las 24 clases", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "02_muestra_por_clase.png"), bbox_inches="tight")
plt.close()

# -------------------------------------------------- 3. variabilidad intra-clase
clases_demo = PRESENT[:6]
fig, axes = plt.subplots(6, 8, figsize=(11, 8.4))
for r, orig in enumerate(clases_demo):
    idxs = np.where(y_train_full == orig)[0][:8]
    for c, idx in enumerate(idxs):
        axes[r, c].imshow(X_train_full[idx].reshape(28, 28), cmap="gray")
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_ylabel(INT2LETRA[orig], fontsize=14, rotation=0, labelpad=14)
fig.suptitle("Variabilidad intra-clase: 8 ejemplos distintos de 6 letras", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "03_variabilidad_intraclase.png"), bbox_inches="tight")
plt.close()

# -------------------------------------------------- 4. histograma de intensidad
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(X_train_full.ravel(), bins=50, color="#4d4d4d")
axes[0].set_title("Histograma de intensidad de píxel (train)")
axes[0].set_xlabel("Valor de píxel (0–255)")
axes[0].set_ylabel("Frecuencia (log)")
axes[0].set_yscale("log")
axes[0].grid(alpha=.3)

axes[1].hist([np.count_nonzero(x) for x in X_train_full], bins=60, color="#1a9850")
axes[1].set_title("Píxeles distintos de cero por imagen")
axes[1].set_xlabel("Píxeles con información (de 784)")
axes[1].set_ylabel("N° de imágenes")
axes[1].grid(alpha=.3)

# imagen media global vs imagen media de la clase A
axes[2].imshow(X_train_full.mean(axis=0).reshape(28, 28), cmap="viridis")
axes[2].set_title("Imagen promedio global (todas las clases)")
axes[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "04_histograma_intensidad.png"), bbox_inches="tight")
plt.close()

# ------------------------------------------- 5. calor de confusas potenciales
# correlación entre los centroides de cada clase: pares más parecidos
centroides = np.stack([X_train_full[y_train_full == c].mean(axis=0) for c in PRESENT])
Cn = centroides / (np.linalg.norm(centroides, axis=1, keepdims=True) + 1e-9)
sim = Cn @ Cn.T
np.fill_diagonal(sim, 0)
pares = []
for i in range(len(PRESENT)):
    for j in range(i + 1, len(PRESENT)):
        pares.append((sim[i, j], INT2LETRA[PRESENT[i]], INT2LETRA[PRESENT[j]]))
pares.sort(reverse=True)
print("\nPares de clases más parecidas (similitud de centroides):")
for s, a, b in pares[:8]:
    print(f"  {a}-{b}: {s:.4f}")

nombres = [INT2LETRA[c] for c in PRESENT]
fig, ax = plt.subplots(figsize=(9, 7.4))
im = ax.imshow(sim, cmap="magma")
ax.set_xticks(range(len(nombres))); ax.set_xticklabels(nombres, fontsize=7)
ax.set_yticks(range(len(nombres))); ax.set_yticklabels(nombres, fontsize=7)
ax.set_title("Similitud de coseno entre imágenes promedio de cada clase\n(pares más altos = señas visualmente parecidas)")
plt.colorbar(im, ax=ax, shrink=.8)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "05_similitud_entre_clases.png"), bbox_inches="tight")
plt.close()

resumen["pares_mas_similares"] = [[a, b, round(float(s), 4)] for s, a, b in pares[:8]]
with open(os.path.join(BASE, "models", "eda_resumen.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
print("\nFiguras generadas en", IMG)
