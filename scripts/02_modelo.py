"""
EP1 TLY1102 - Modelamiento MLP sobre Sign Language MNIST
Variante propia: subconjunto de 11 letras (A-L sin J) + comparacion de arquitecturas
Genera: models/*.joblib, models/metricas.json, images/06..11
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, top_k_accuracy_score)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, IMG, MODELS = (os.path.join(BASE, d) for d in ("data", "images", "models"))
os.makedirs(MODELS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
SEED = 42

LETRA = {i: chr(65 + i) for i in range(26)}
PRESENT = sorted(set(range(26)) - {9, 25})          # 24 clases reales del dataset
CLASES_VAR = [c for c in PRESENT if c <= 11]          # A..L sin J  -> 11 clases
print("Clases 12:", [LETRA[c] for c in CLASES_VAR])

# ------------------------------------------------------------------ carga
tr = pd.read_csv(os.path.join(DATA, "sign_mnist_train.csv"))
te = pd.read_csv(os.path.join(DATA, "sign_mnist_test.csv"))
X_raw_tr, y_raw_tr = tr.drop("label", axis=1).values.astype(np.float64), tr["label"].values
X_raw_te, y_raw_te = te.drop("label", axis=1).values.astype(np.float64), te["label"].values


def preparar(clases):
    """Filtra las clases pedidas, remapea etiquetas a 0..K-1 y normaliza a [0,1]."""
    m_tr = np.isin(y_raw_tr, clases)
    m_te = np.isin(y_raw_te, clases)
    remap = {orig: i for i, orig in enumerate(clases)}
    Xtr, ytr = X_raw_tr[m_tr], np.array([remap[v] for v in y_raw_tr[m_tr]])
    Xte, yte = X_raw_te[m_te], np.array([remap[v] for v in y_raw_te[m_te]])
    return Xtr / 255.0, ytr, Xte / 255.0, yte


def entrenar(Xtr, ytr, hidden, nombre, epochs_max=200):
    """Entrena un MLP con early stopping y devuelve (modelo, historial, segundos)."""
    print(f"\n>>> Entrenando {nombre}: hidden={hidden}")
    t0 = time.time()
    m = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",          # no saturacion en capas ocultas
        solver="adam",              # descenso de gradiente adaptativo
        alpha=1e-4,                 # regularizacion L2 suave
        batch_size=200,             # mini-batch
        learning_rate_init=1e-3,
        max_iter=epochs_max,
        early_stopping=True,        # valida 10% interno y corta si no mejora
        validation_fraction=0.10,
        n_iter_no_change=10,
        random_state=SEED,
    )
    m.fit(Xtr, ytr)
    seg = time.time() - t0
    print(f"    epochs={m.n_iter_}  loss_final={m.loss_curve_[-1]:.4f}  tiempo={seg:.1f}s")
    return m, seg


def evaluar(m, Xte, yte, nombres, prefijo, grados_confusion=8):
    """Metricas completas + matriz de confusion + figuras."""
    yp = m.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec, rec, f1, sup = precision_recall_fscore_support(yte, yp, average="macro", zero_division=0)
    rep = classification_report(yte, yp, target_names=nombres, output_dict=True, zero_division=0)

    print(f"    accuracy={acc:.4f}  precision_macro={prec:.4f}  recall_macro={rec:.4f}  f1_macro={f1:.4f}")

    cm = confusion_matrix(yte, yp)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # --- matriz de confusion ---
    n = len(nombres)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.55), max(5, n * 0.5)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(nombres, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(nombres, fontsize=8)
    ax.set_xlabel("Prediccion del modelo"); ax.set_ylabel("Letra real")
    ax.set_title(f"Matriz de confusion normalizada\n{prefijo}  (accuracy={acc:.3f})")
    for i in range(n):
        for j in range(n):
            if cmn[i, j] > 0.005:
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=.85, label="proporcion de la clase real")
    plt.tight_layout(); plt.savefig(os.path.join(IMG, f"07_matriz_confusion_{prefijo}.png"), bbox_inches="tight")
    plt.close()

    # --- pares mas confundidos ---
    fuera = cm.copy(); np.fill_diagonal(fuera, 0)
    pares = []
    for i in range(n):
        for j in range(n):
            if fuera[i, j] > 0:
                pares.append((int(fuera[i, j]), nombres[i], nombres[j], round(float(cmn[i, j]) * 100, 1)))
    pares.sort(reverse=True)
    print("    Top confesiones real->predicha:", pares[:5])

    return {"accuracy": float(acc), "precision_macro": float(prec), "recall_macro": float(rec),
            "f1_macro": float(f1), "report": rep, "cm": cm.tolist(),
            "pares_confundidos": pares[:grados_confusion],
            "n_test": int(len(yte)), "y_pred": yp}


resultados = {}
nombres_var = [LETRA[c] for c in CLASES_VAR]
nombres24 = [LETRA[c] for c in PRESENT]

# ============================ M1: 11 letras, 1 capa oculta =================
Xtr, ytr, Xte, yte = preparar(CLASES_VAR)
Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.15, random_state=SEED, stratify=ytr)
print(f"\nParticion: train={Xfit.shape} val={Xval.shape} test_holdout={Xte.shape}")

m1, t1 = entrenar(Xfit, yfit, (128,), "M1_11letras_1capa")
r1 = evaluar(m1, Xte, yte, nombres_var, "M1_11letras_1capa")
r1["tiempo_s"], r1["epochs"] = round(t1, 1), int(m1.n_iter_)
r1["val_accuracy_interno"] = float(max(m1.validation_scores_))
r1["loss_curve"] = [float(v) for v in m1.loss_curve_]
r1["val_scores"] = [float(v) for v in m1.validation_scores_]
joblib.dump({"model": m1, "clases": CLASES_VAR, "nombres": nombres_var}, os.path.join(MODELS, "m1_11letras_1capa.joblib"))
resultados["M1_11letras_1capa"] = r1

# ============================ M2: 11 letras, 2 capas ocultas ===============
m2, t2 = entrenar(Xfit, yfit, (256, 128), "M2_11letras_2capas")
r2 = evaluar(m2, Xte, yte, nombres_var, "M2_11letras_2capas")
r2["tiempo_s"], r2["epochs"] = round(t2, 1), int(m2.n_iter_)
r2["val_accuracy_interno"] = float(max(m2.validation_scores_))
r2["loss_curve"] = [float(v) for v in m2.loss_curve_]
r2["val_scores"] = [float(v) for v in m2.validation_scores_]
joblib.dump({"model": m2, "clases": CLASES_VAR, "nombres": nombres_var}, os.path.join(MODELS, "m2_11letras_2capas.joblib"))
resultados["M2_11letras_2capas"] = r2

# ============================ M3: 24 letras, 1 capa oculta =================
Xtr24, ytr24, Xte24, yte24 = preparar(PRESENT)
Xfit24, Xval24, yfit24, yval24 = train_test_split(Xtr24, ytr24, test_size=0.15, random_state=SEED, stratify=ytr24)
m3, t3 = entrenar(Xfit24, yfit24, (128,), "M3_24letras_1capa")
r3 = evaluar(m3, Xte24, yte24, nombres24, "M3_24letras_1capa")
r3["tiempo_s"], r3["epochs"] = round(t3, 1), int(m3.n_iter_)
r3["loss_curve"] = [float(v) for v in m3.loss_curve_]
r3["val_scores"] = [float(v) for v in m3.validation_scores_]
joblib.dump({"model": m3, "clases": PRESENT, "nombres": nombres24}, os.path.join(MODELS, "m3_24letras_1capa.joblib"))
resultados["M3_24letras_1capa"] = r3

# ============ M4: robustez de M1 ante traslacion (limitacion del MLP) =======
# El MLP no tiene invarianza a traslacion: desplazamos el test 2 px y medimos la caida.
desplazamientos = {}
for k in (1, 2, 3):
    Xt = np.roll(Xte.reshape(-1, 28, 28), shift=k, axis=2).reshape(len(Xte), -1)
    acc_k = accuracy_score(yte, m1.predict(Xt))
    desplazamientos[k] = float(acc_k)
    print(f"    M1 con test desplazado {k} px -> accuracy={acc_k:.4f}")
resultados["M4_robustez_traslacion"] = {
    "base_accuracy": r1["accuracy"], "desplazamientos_px": desplazamientos,
    "caida_relativa": {k: round((r1["accuracy"] - v) / r1["accuracy"] * 100, 2) for k, v in desplazamientos.items()},
}

# ============================ figuras comparativas =========================
# curvas de aprendizaje
NOMBRE_MODELO = {
    "M1_11letras_1capa": "M1 · 11 letras, 1 capa oculta (128)",
    "M2_11letras_2capas": "M2 · 11 letras, 2 capas ocultas (256-128)",
    "M3_24letras_1capa": "M3 · 24 letras, 1 capa oculta (128)",
    "M4_robustez_traslacion": "M4 · robustez a la traslación",
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (k, r) in zip(axes, [("M1_11letras_1capa", r1), ("M2_11letras_2capas", r2), ("M3_24letras_1capa", r3)]):
    ep = range(1, len(r["loss_curve"]) + 1)
    ax.plot(ep, r["loss_curve"], color="#2c7bb6", lw=2, label="Perdida en entrenamiento")
    ax2 = ax.twinx()
    if r["val_scores"]:
        ax2.plot(range(1, len(r["val_scores"]) + 1), r["val_scores"], color="#d7191c", lw=2, ls="--",
                 label="Accuracy de validacion")
    ax2.set_ylim(0, 1); ax2.set_ylabel("Accuracy validacion", color="#d7191c")
    ax.set_xlabel("Epoca"); ax.set_ylabel("Perdida (entropia cruzada)", color="#2c7bb6")
    ax.set_title(f"{NOMBRE_MODELO[k]}\nAccuracy en prueba = {r['accuracy']:.3f} · {r['epochs']} epocas",
                 fontsize=10)
    ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "06_curvas_entrenamiento.png"), bbox_inches="tight"); plt.close()

# comparacion de modelos
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ks = list(resultados.keys())[:3]
met = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
x = np.arange(len(ks)); w = 0.2
for i, mm in enumerate(met):
    axes[0].bar(x + i * w, [resultados[k][mm] for k in ks], w, label=mm)
axes[0].set_xticks(x + 1.5 * w)
axes[0].set_xticklabels(["M1\n11 letras\n1 capa", "M2\n11 letras\n2 capas", "M3\n24 letras\n1 capa"], fontsize=8)
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.3)
axes[0].set_title("Metricas en el conjunto de prueba (hold-out)")

axes[1].bar(["base\n(0 px)"] + [f"{k} px" for k in desplazamientos], 
            [r1["accuracy"]] + list(desplazamientos.values()), color="#d7191c", alpha=.75)
axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=.3)
axes[1].set_title("M1: degradacion al desplazar la imagen\n(limitacion: el MLP no es invariante a traslacion)")
for i, v in enumerate([r1["accuracy"]] + list(desplazamientos.values())):
    axes[1].text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "09_comparacion_modelos.png"), bbox_inches="tight"); plt.close()

# ============================ analisis de errores ==========================
yp = np.array(r1["y_pred"])
mal = np.where(yp != yte)[0]
bien = np.where(yp == yte)[0]
rng = np.random.default_rng(SEED)
prob = m1.predict_proba(Xte)

fig, axes = plt.subplots(4, 6, figsize=(11, 8))
sel_bien = rng.choice(bien, 12, replace=False)
sel_mal = rng.choice(mal, 12, replace=False)
for c in range(6):
    i = sel_bien[c]
    axes[0, c].imshow(Xte[i].reshape(28, 28), cmap="gray"); axes[0, c].axis("off")
    axes[0, c].set_title(f"OK: {nombres_var[yp[i]]}", fontsize=8, color="green")
    i = sel_bien[c + 6]
    axes[1, c].imshow(Xte[i].reshape(28, 28), cmap="gray"); axes[1, c].axis("off")
    axes[1, c].set_title(f"OK: {nombres_var[yp[i]]}", fontsize=8, color="green")
    i = sel_mal[c]
    axes[2, c].imshow(Xte[i].reshape(28, 28), cmap="gray"); axes[2, c].axis("off")
    conf = prob[i, yp[i]] * 100
    axes[2, c].set_title(f"real {nombres_var[yte[i]]}\npred {nombres_var[yp[i]]} ({conf:.0f}%)", fontsize=7, color="red")
    i = sel_mal[c + 6]
    axes[3, c].imshow(Xte[i].reshape(28, 28), cmap="gray"); axes[3, c].axis("off")
    conf = prob[i, yp[i]] * 100
    axes[3, c].set_title(f"real {nombres_var[yte[i]]}\npred {nombres_var[yp[i]]} ({conf:.0f}%)", fontsize=7, color="red")
fig.text(0.5, 0.975, "Aciertos (verde, filas 1-2) y errores (rojo, filas 3-4) - modelo M1",
         ha="center", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.945])
plt.savefig(os.path.join(IMG, "08_analisis_errores.png"), bbox_inches="tight"); plt.close()

# confianza en aciertos vs errores (muy util para hablar de calibracion)
aciertos_conf = prob[bien][np.arange(len(bien)), yp[bien]] if len(bien) else np.array([])
errores_conf = prob[mal][np.arange(len(mal)), yp[mal]] if len(mal) else np.array([])
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(aciertos_conf, bins=25, alpha=.65, label=f"Aciertos (n={len(bien)})", color="#1a9850")
ax.hist(errores_conf, bins=25, alpha=.65, label=f"Errores (n={len(mal)})", color="#d7191c")
ax.set_xlabel("Probabilidad asignada a la clase predicha"); ax.set_ylabel("N° de imagenes")
ax.set_title("Confianza del modelo: en los errores se concentra por debajo de 0.6")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "10_confianza_errores.png"), bbox_inches="tight"); plt.close()

print(f"\nAciertos: {len(bien)}  Errores: {len(mal)}")
print(f"Confianza media en aciertos: {aciertos_conf.mean():.3f} | en errores: {errores_conf.mean():.3f}")
# ejemplos concretos para la diapositiva
ejemplos_error = []
for i in sel_mal:
    ejemplos_error.append({
        "real": nombres_var[yte[i]], "predicho": nombres_var[yp[i]],
        "confianza": round(float(prob[i, yp[i]]) * 100, 1),
        "segunda_opcion": nombres_var[int(np.argsort(prob[i])[::-1][1])],
    })

resumen = {
    "semilla": SEED,
    "particion": {"train": int(len(Xfit)), "val": int(len(Xval)), "test_holdout": int(len(Xte))},
    "variante": "11 letras A-L (sin J)",
    "modelos": {k: {kk: vv for kk, vv in v.items() if kk not in ("report", "cm", "y_pred")} for k, v in resultados.items()},
    "classification_report_M1": r1["report"],
    "matriz_confusion_M1": r1["cm"],
    "matriz_confusion_M3": r3["cm"],
    "ejemplos_error_M1": ejemplos_error,
    "confianza_media": {"aciertos": round(float(aciertos_conf.mean()), 4),
                        "errores": round(float(errores_conf.mean()), 4)},
    "aciertos": int(len(bien)), "errores": int(len(mal)),
}
with open(os.path.join(MODELS, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
print("\n=== TERMINADO ===")
print(json.dumps({k: round(v["accuracy"], 4) for k, v in resultados.items() if "accuracy" in v}, indent=2))
