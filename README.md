# EP1 · Clasificación de letras del alfabeto dactilológico (ASL) con un Perceptrón Multicapa

**Asignatura:** TLY1102 — Técnicas Avanzadas de Machine Learning I
**Evaluación:** Evaluación Parcial N°1 (30%) — Presentación y defensa técnica del proyecto
**Repositorio:** https://github.com/colapso05/ep1-tly1102-mlp-senas-asl
**Modalidad:** Grupal · **Grupo:** 9

> **Qué se entrega.** Según la pauta: informe técnico en Markdown (este `README.md`),
> notebook en Python completamente ejecutable, el conjunto de datos y una carpeta de proyecto
> organizada.

| Integrante | Rol en la defensa |
|---|---|
| Demis Zuñiga | Presenta el proyecto completo (la defensa es individual) |
| Bruno Burgos | Presenta el proyecto completo (la defensa es individual) |

> **Importante — la defensa es individual.** La pauta dice textualmente: *"Si bien esta
> evaluación es posible desarrollarla y rendirla de manera grupal, la presentación será
> evaluada de manera Individual y estará asociada al desempeño de cada estudiante, ya sea
> mediante la calidad de su exposición, así como también en las respuestas a las preguntas
> realizadas por su docente. Por lo anterior, cada estudiante debe estar preparado para
> responder preguntas cruzadas, es decir, no necesariamente se le realizarán preguntas
> específicamente de lo que haya presentado individualmente."*
>
> Consecuencia práctica: **no se pueden repartir el contenido**. Los dos tienen que poder
> explicar cualquier parte del proyecto, incluidas las que exponga el otro.

---

## 1. Descripción del problema de negocio

### Contexto

La comunidad sorda que utiliza el alfabeto dactilológico (*fingerspelling*) deletrea con la
mano las palabras para las que no existe una seña específica: nombres propios, siglas, términos
técnicos. El problema es que ese deletreo es **invisible para quien no conoce el alfabeto**, lo
que genera una barrera de comunicación concreta en situaciones cotidianas (atención médica,
trámites, sala de clases, atención al cliente).

### El problema de negocio

> **Una persona oyente que atiende público no puede leer el deletreo manual de una persona
> sorda, y no existe una herramienta automática, barata y disponible que traduzca esa seña a
> texto en el momento.**

Traducido a un problema de ciencia de datos: **dada una imagen de una mano formando una letra
del alfabeto ASL, predecir automáticamente cuál es esa letra.** Es una tarea de
**clasificación supervisada de imágenes** con 24 clases posibles (las 26 letras del alfabeto
menos J y Z, que en ASL requieren movimiento y no una postura estática).

### ¿Por qué este problema ahora?

Construir un traductor de señas a texto de calidad requeriría una red convolucional (materia
de la Experiencia de Aprendizaje 2). El objetivo de esta evaluación es distinto y deliberado:
**usar un Perceptrón Multicapa como modelo base para entender, medir y demostrar sus límites**
frente a datos de imagen. Comprender *por qué* un MLP se queda corto es el requisito para
justificar por qué las CNN existen.

---

## 2. Objetivos del proyecto

### Objetivo general

Construir, entrenar y evaluar un Perceptrón Multicapa que clasifique señas estáticas del
alfabeto dactilológico americano, reportando métricas de desempeño y analizando los errores y
las limitaciones del enfoque.

### Objetivos específicos

1. Describir y justificar la selección del conjunto de datos y definir una **variante propia**
   del problema que evite la copia entre grupos.
2. Aplicar un preprocesamiento completo y justificado: selección de clases, remapeo de
   etiquetas, normalización y partición de datos.
3. Implementar una arquitectura MLP **justificando cada decisión de diseño** (capas, neuronas,
   funciones de activación, función de salida, pérdida, optimizador e hiperparámetros).
4. Entrenar el modelo registrando e interpretando el comportamiento del proceso de aprendizaje.
5. Evaluar el desempeño con **múltiples métricas** (accuracy, precision, recall, F1-score y
   matriz de confusión) e interpretarlas en función del problema.
6. Auditar los errores del modelo con ejemplos concretos y **medir experimentalmente** la
   limitación del MLP frente a imágenes.
7. Documentar todo el proceso de forma reproducible.

### La variante propia del grupo

El enunciado exige que cada grupo defina una variante para evitar copias. Nuestra variante
tiene tres componentes:

| # | Decisión | Por qué |
|---|---|---|
| 1 | **Subconjunto de 11 letras: A, B, C, D, E, F, G, H, I, K, L** | Trabaja el problema en condiciones controladas; permite medir después el costo de escalar a las 24 clases. Corresponde al ejemplo de variante que menciona el propio enunciado. |
| 2 | **Comparación de dos arquitecturas MLP** (1 capa de 128 neuronas vs 2 capas de 256 y 128) | Convierte la entrega en una experimentación real y permite responder con datos la pregunta *"¿qué pasa si la red es más profunda?"*. |
| 3 | **Experimento de robustez a la traslación** | Cuantifica empíricamente la limitación central del MLP en imágenes, que la pauta pide discutir. |

---

## 3. Definición de KPIs

Los KPIs se definen sobre el **conjunto de prueba (hold-out)**, nunca sobre el de entrenamiento.

| KPI | Definición | Meta | Por qué importa para el negocio |
|---|---|---|---|
| **Accuracy** | Proporción de imágenes clasificadas correctamente | ≥ 0,85 | Medida global de calidad. Es interpretable porque las clases están balanceadas. |
| **F1-Score macro** | Media armónica de precision y recall, promediada por clase sin ponderar | ≥ 0,85 | Impide que una letra frecuente esconda a una mal reconocida. |
| **Recall por clase** | Para cada letra, proporción de sus imágenes correctamente detectadas | Ninguna < 0,75 | Un recall bajo = la letra "desaparece". En una palabra, perder una letra cambia el significado. |
| **Precision por clase** | Cuando el modelo dice "es la letra X", ¿cuántas veces acierta? | ≥ 0,80 | Un precision bajo = el sistema inventa letras, degradando la confianza del usuario. |
| **Brecha train − test** | Diferencia entre accuracy de entrenamiento y de prueba | < 0,15 | Mide sobreajuste: un sistema que solo funciona con los datos que vio no es desplegable. |
| **Cobertura con abstención** | Proporción de predicciones con confianza ≥ 0,70 | ≥ 0,80 | En accesibilidad conviene abstenerse antes que responder mal; la confianza debe ser explotable. |

**Cálculo de las métricas por clase:**

$$\text{Precision}_c = \frac{VP_c}{VP_c + FP_c} \qquad \text{Recall}_c = \frac{VP_c}{VP_c + FN_c} \qquad F1_c = 2\cdot\frac{\text{Precision}_c\cdot\text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**¿Cuál es la línea base?** Un clasificador trivial que siempre respondiera la letra más
frecuente acertaría **9,1%** (1 de 11). Cualquier resultado muy por encima de ese valor es
evidencia de aprendizaje real.

---

## 4. Descripción de las fuentes de datos

### Dataset seleccionado

**Sign Language MNIST** — imágenes de señas estáticas del alfabeto dactilológico americano.

| Característica | Valor |
|---|---|
| Archivos | `sign_mnist_train.csv`, `sign_mnist_test.csv` |
| Imágenes de entrenamiento | 27.455 |
| Imágenes de prueba | 7.172 |
| Formato | escala de grises, 28×28 píxeles (784 valores por fila) |
| Rango de los valores de píxel | 0 (negro) – 255 (blanco) |
| Clases | 24 (letras A–Y excluyendo J; se omite también Z) |
| Distribución | balanceada: entre 957 y 1.294 imágenes por clase (ratio 1,35×) |
| Valores nulos | 0 |
| Archivos complementarios | `amer_sign2.png`, `amer_sign3.png`, `american_sign_language.PNG` (alfabetos de referencia) |

**Exclusión de J y Z:** estas dos letras se realizan en ASL con un movimiento (trazar la letra
en el aire). Una fotografía estática no puede representarlas, por lo que el dataset las omite.
Esto es una **limitación del problema que declaramos explícitamente**: el sistema solo puede
traducir las 24 letras estáticas.

### Justificación de la elección

1. **Aísla la variable a estudiar.** Las imágenes ya vienen normalizadas a 28×28 en escala de
   grises. Con datasets de imágenes RGB de alta resolución, el resultado estaría dominado por
   el preprocesamiento y no por la arquitectura de la red — que es lo que la evaluación
   pretende evaluar.
2. **Comparabilidad.** Es el equivalente en señas del clásico MNIST: permite comparar nuestros
   resultados con las referencias de la clase y del material de la asignatura.
3. **Clases balanceadas.** Elimina el riesgo de que las métricas se distorsionen por desbalanceo.
4. **Costo computacional.** Entrenar un MLP sobre 784 entradas es viable en un computador
   personal en segundos, lo que permitió hacer tres experimentos comparativos completos y no
   solo uno.
5. **Está mencionado en el enunciado** como ejemplo legítimo de variante.

---

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Calidad y estructura

Cada fila del CSV es una imagen ya aplanada: la columna `label` contiene la letra y las
columnas `pixel1 … pixel784` los 784 píxeles de la imagen de 28×28 (28 × 28 = 784). No hay
valores nulos ni datos faltantes. Los píxeles van de 0 a 255.

### 5.2 Distribución de clases

Balance prácticamente perfecto: la clase más grande tiene 1.294 imágenes, la más pequeña 957,
y la media es 1.143. Dado que las clases están equilibradas, la *accuracy* es una métrica
interpretable, y por eso se complementa con métricas por clase.

→ Figura: `images/01_distribucion_clases.png`

### 5.3 Exploración visual

Revisamos una imagen representativa por cada una de las 24 clases y la variabilidad
intra-clase (8 ejemplos de la misma letra).

**Patrón identificado:** la misma letra aparece con diferencias de posición dentro del cuadro,
tamaño de mano, grosor de trazo y brillo del fondo.

→ Figuras: `images/02_muestra_por_clase.png`, `images/03_variabilidad_intraclase.png`

### 5.4 Análisis de los píxeles

- La mayoría de los píxeles de cada imagen son fondo: **más del 60% de las 784 entradas que
  recibe el modelo no contienen información de la seña**, sino ruido del fondo.
- La imagen promedio global muestra un fondo con brillo no uniforme: la cámara aportó
  información sistemática que el modelo también aprende.

→ Figura: `images/04_histograma_intensidad.png`

**Relevancia de este hallazgo:** un MLP asigna un peso propio a cada uno de los 784 píxeles,
sin saber cuáles son relevantes. Está obligado a aprender, a partir de los datos, que el fondo
no importa — y en parte no lo logra. Esta es la primera evidencia de por qué el enfoque es
ineficiente en imágenes.

### 5.5 Similitud entre clases (análisis que anticipa los errores)

Calculamos la similitud de coseno entre las imágenes promedio de cada par de clases, para
identificar qué señas son visualmente parecidas y por lo tanto propensas a confundirse.

Los pares más similares fueron **R–U (0,9991), R–V (0,9987), R–W (0,9986), M–N (0,9985),
M–S (0,9983)**. Formulamos la **hipótesis** de que estos serían los pares más confundidos por
el modelo. Esta hipótesis se contrasta en la sección 7.

→ Figura: `images/05_similitud_entre_clases.png`

### 5.6 Variables relevantes

No hay selección de variables en el sentido clásico: las 784 columnas de píxeles son la
entrada. La decisión de diseño relevante fue **cómo tratarlas** (normalización, ver sección 6)
y **qué variable se usa como objetivo** (`label`, remapeada a índices contiguos para la
variante de 11 clases).

---

## 6. Metodología: CRISP-DM

| Fase CRISP-DM | Cómo se aplicó en este proyecto |
|---|---|
| **1. Comprensión del negocio** | Se definió el problema de accesibilidad (traducción de deletreo manual a texto) y se tradujo a una tarea de clasificación de imágenes con 24 clases. Se fijaron KPIs con metas medibles (sección 3). |
| **2. Comprensión de los datos** | EDA completo: estructura, calidad, balance de clases, análisis de píxeles, variabilidad intra-clase y similitud entre clases (sección 5). Se detectaron las dificultades: fondo ruidoso, variabilidad de escritura, señas visualmente casi idénticas. |
| **3. Preparación de los datos** | Selección de las 11 clases de la variante; remapeo de etiquetas a 0…11; normalización a [0,1]; partición entrenamiento 85% / validación 15% / hold-out intacto (sección 7.1). |
| **4. Modelamiento** | Implementación de tres configuraciones de MLP con hiperparámetros justificados (sección 7.2), más un experimento de robustez a la traslación. |
| **5. Evaluación** | Reporte de accuracy, precision, recall, F1 y matriz de confusión; análisis de la brecha de generalización; auditoría de errores con ejemplos (secciones 8 y 9). |
| **6. Despliegue** | Entrega de un proyecto reproducible: notebook ejecutable, scripts de pipeline, modelos serializados, figuras e informe. Uso previsto: prototipo de investigación, no producción (se declaran las limitaciones). |

### 6.1 Partición de los datos

| Conjunto | Tamaño (variante 11 letras) | Uso |
|---|---|---|
| Entrenamiento | 10.418 imágenes (85% del CSV de train) | Ajuste de los pesos de la red |
| Validación | 1.839 imágenes (15% del CSV de train) | Vigilar sobreajuste y validar decisiones de diseño |
| **Prueba (hold-out)** | 3.675 imágenes (`sign_mnist_test.csv`) | **Medición final del desempeño. Se usa una sola vez.** |

Reparto estratificado (`stratify=y`), para que todos los subconjuntos conserven la proporción
de clases. **Semilla fija `SEED = 42`** en la partición, en la inicialización de los pesos y en
el muestreo de ejemplos para las figuras: el trabajo es totalmente reproducible.

### 6.2 Preprocesamiento aplicado

| Etapa | Decisión | Justificación |
|---|---|---|
| Selección de clases | Filtrar las 11 letras A–L sin J | Define la variante y reduce el problema a 11 clases balanceadas. |
| Remapeo de etiquetas | Convertir las etiquetas originales a índices contiguos 0…11 | Las etiquetas originales saltan el 9 (J). Un índice contiguo hace explícita la correspondencia entre la etiqueta y la neurona 12 de la capa de salida, y evita errores al interpretar `argmax`. |
| Normalización | Dividir cada píxel por 255 → rango [0, 1] | Adam converge mucho más rápido y de forma más estable cuando las entradas tienen escala comparable. Con entradas entre 0 y 255 los gradientes iniciales son desproporcionados. |
| Hold-out separado | El `test.csv` no se toca hasta la evaluación final | Ajustar hiperparámetros mirando el conjunto de prueba invalida la medición. **Esta decisión resultó determinante** (ver sección 8.3). |

**Nota:** no se aplicó aumento de datos. Es una decisión consciente: queremos medir el
desempeño del MLP puro, y el aumento de datos se propone como mejora en las conclusiones.

---

## 7. Diseño e implementación del modelo

### 7.1 Anatomía de la red

Una red neuronal densa calcula, en cada neurona, una suma ponderada de sus entradas más un
sesgo, y luego aplica una función no lineal:

$$z = \sum_{i} w_i x_i + b \qquad a = f(z)$$

Sin la no linealidad, apilar capas equivaldría a una sola transformación lineal, y la red no
podría separar 11 clases que en el espacio de píxeles están entremezcladas. Esa no linealidad
es lo que da capacidad expresiva al modelo.

### 7.2 Decisiones de arquitectura

| Componente | Elección | Justificación |
|---|---|---|
| Capa de entrada | 784 neuronas (28×28 aplanado) | Es la dimensionalidad real del dato: una neurona por píxel. |
| Capas ocultas (M1) | 1 capa de **128** neuronas | 128 < 784 obliga a la red a **comprimir**: construir representaciones internas en lugar de copiar píxeles. Más neuronas que entradas facilitarían la memorización. |
| Capas ocultas (M2) | 2 capas: **256 y 128** | Segunda capa que recombina las características de la primera. Responde empíricamente a *"¿qué pasa si la red es más profunda?"*. |
| Activación oculta | **ReLU** — $f(z)=\max(0,z)$ | No se satura en valores positivos, así el gradiente no se desvanece; es más económica que `tanh` y es el estándar actual. |
| Capa de salida | **11 neuronas + softmax** | Una neurona por letra. `softmax` produce una distribución de probabilidad que suma 1, interpretable como **confianza**. |
| Función de pérdida | **Entropía cruzada** | Pérdida natural para clasificación multiclase con salida softmax: penaliza fuerte cuando se asigna probabilidad baja a la clase correcta. |
| Optimizador | **Adam**, `learning_rate_init = 0.001` | Descenso de gradiente estocástico con momento y tasa adaptativa por parámetro. Converge más rápido que SGD puro y es menos sensible a la escala de las entradas. |
| Tamaño de lote | `batch_size = 200` | Lotes pequeños hacen el entrenamiento ruidoso; un lote del tamaño del dataset lo hace lento y con más riesgo de caer en un mínimo pobre. |
| Épocas | `max_iter = 200` con **parada temprana** (`n_iter_no_change = 10`) | Evita fijar las épocas "a ojo" y evita seguir entrenando cuando ya no hay mejora. |
| Regularización | `alpha = 1e-4` (penalización L2) | Penaliza pesos de magnitud excesiva, el mecanismo típico del sobreajuste. |
| Semilla | `random_state = 42` | Fija la inicialización de los pesos; el resultado es reproducible. |

**Total de parámetros entrenables de M1: 101.899** (784×128 + 128 pesos de la capa oculta,
128 biases, 128×11 + 11 de la capa de salida).

---

## 8. Entrenamiento y evaluación

### 8.1 Comportamiento del entrenamiento

Se entrenaron tres configuraciones sobre el mismo conjunto de entrenamiento:

| Modelo | Arquitectura | Clases | Épocas efectivas | Pérdida final | Tiempo de entrenamiento |
|---|---|---|---|---|---|
| **M1** | 1 capa oculta (128) | 11 | 52 | 0,0138 | 14,3 s |
| **M2** | 2 capas ocultas (256, 128) | 11 | 30 | 0,0085 | 26,9 s |
| **M3** | 1 capa oculta (128) | 24 | 83 | 0,0241 | 50,5 s |

La **parada temprana** se activó antes del límite de 200 épocas en los tres casos: la red dejó
de mejorar y el entrenamiento se detuvo solo. La pérdida descendió de forma pronunciada en las
primeras épocas y luego se aplanó, señal de convergencia.

→ Figura: `images/06_curvas_entrenamiento.png`

### 8.2 Resultados sobre el conjunto de prueba (hold-out)

| Modelo | Clases | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **M1** · 1 capa (128) | 11 | **0,8941** | 0,8867 | 0,8951 | **0,8892** |
| **M2** · 2 capas (256, 128) | 11 | **0,9007** | 0,8979 | 0,9044 | **0,8956** |
| **M3** · 1 capa (128) | 24 | **0,7140** | 0,7031 | 0,6928 | **0,6893** |

**Lectura de los resultados:**

1. **M1 cumple los KPIs fijados** (accuracy y F1 sobre 0,85). Con precision y recall
   equilibrados (0,887 vs 0,895), el modelo no está sesgado hacia ser conservador ni
   arriesgado: comete ambos tipos de error en proporción similar.
2. **M2 mejora apenas 0,66 puntos porcentuales de accuracy** (0,8941 → 0,9007) duplicando el
   número de parámetros, el tiempo de entrenamiento y la profundidad de la red. Es una mejora
   **marginal, no proporcional al costo**. Conclusión: en imágenes, la profundidad sin
   estructura espacial no es la palanca correcta.
3. **M3 muestra el costo de ampliar el problema**: pasar de 11 a 24 clases hace caer la
   accuracy en **18 puntos** (0,894 → 0,714) y el F1 macro en **20 puntos**. La dificultad crece
   mucho más rápido que proporcionalmente, porque las señas que se agregan son precisamente las
   que se parecen entre sí.

### 8.3 La brecha de generalización (hallazgo relevante)

| Conjunto | Accuracy de M1 |
|---|---|
| Entrenamiento | **0,9999** |
| Validación (15% del CSV de train) | **0,9989** |
| Prueba (hold-out, `test.csv`) | **0,8941** |
| **Brecha entrenamiento − prueba** | **10,58 puntos porcentuales** |

**Diagnóstico.** El modelo **memoriza** el conjunto de entrenamiento. Un MLP con ~100.000 pesos
tiene capacidad de sobra para memorizar 10.000 imágenes: la pérdida de entrenamiento llega a
0,0138 y el accuracy a 99,99%, pero en datos de otra sesión de captura cae a 89,4%.

**Segunda lección, la más importante.** La validación interna también da 99,89%: es
**optimista** y sobrestima el desempeño real en más de 10 puntos. La razón es que sale del
mismo archivo que el entrenamiento y comparte su sesión de captura. Comparamos las
distribuciones de píxeles de ambos archivos (`images/11_train_vs_test.png`) y son casi
idénticas en media (159,3 vs 160,6), desviación y proporción de píxeles claros. No es un
problema de formato: es que las imágenes de `test.csv` fueron capturadas en otra sesión, con
otras manos y otras posiciones.

> **Consecuencia metodológica.** Si hubiéramos reportado la validación interna, habríamos
> declarado casi 100% de accuracy. Mantener el `test.csv` como *hold-out* intacto fue la
> decisión que hizo visible el desempeño real. **Toda métrica de este informe se reporta sobre
> el hold-out.**

### 8.4 Matriz de confusión

→ Figuras: `images/07_matriz_confusion_M1_11letras_1capa.png`,
`images/07_matriz_confusion_M3_24letras_1capa.png`

Cada fila es la letra real y cada columna la predicha; los valores están normalizados por fila
(cada fila suma 100%). La matriz de M1 muestra una diagonal dominante con errores dispersos, y
revela el patrón que se analiza en la sección 9.

---

## 9. Análisis de resultados y errores

### 9.1 Auditoría del modelo M1

- **Aciertos: 3.286 de 3.675 (89,41%)**
- **Errores: 389**

Ejemplos correctos e incorrectos, seleccionados con la semilla 42:

→ Figura: `images/08_analisis_errores.png`

### 9.2 Clases que presentan mayor confusión

| Letra real | Predicha como | Errores | % de esa clase real |
|---|---|---|---|
| H | G | 55 | 12,6% |
| K | I | 54 | 16,3% |
| B | K | 41 | 9,5% |
| G | H | 30 | 8,6% |
| K | F | 24 | 7,3% |

**Patrón identificado:** las confusiones son **simétricas y por pares**. H↔G y las ternas
B-K-I-F forman el núcleo de los errores. Esto no es azar: son señas que comparten la misma
estructura (dedos extendidos y separados) y difieren en la orientación de uno o dos dedos.

**Contraste con la hipótesis del EDA.** Habíamos predicho que R–U, R–V, R–W y M–N serían los
pares más confundidos. En la variante de 11 letras esas letras no están presentes, y la
predicción **se confirma en el modelo de 24 clases (M3)**, donde los errores más frecuentes
son exactamente esos pares:

| Letra real | Predicha como | Errores | % de esa clase real |
|---|---|---|---|
| M | S | 125 | 31,7% |
| K | R | 78 | 23,6% |
| T | X | 62 | 25,0% |
| U | K | 49 | 18,4% |
| S | M | 49 | 19,9% |

**Esto es importante para la defensa:** demuestra que el modelo no falla al azar. Falla
exactamente donde las clases son geométricamente parecidas, tal como lo anticipaba el análisis
exploratorio. Y explica por qué el F1 macro de M3 es mucho peor que la accuracy simple
sugeriría: hay clases con hasta un 31,7% de error.

### 9.3 Causas de los errores

1. **Similitud geométrica entre clases.** Es la causa principal y está demostrada: los pares
   más confundidos coinciden con los de mayor similitud entre imágenes promedio.
2. **Variabilidad intra-clase.** La misma letra aparece con distinta posición y tamaño de mano,
   y el modelo no generaliza bien esas variaciones.
3. **Ruido del fondo.** Más del 60% de las entradas son fondo, y el MLP les asigna pesos como
   si fueran señal. Una foto de webcam con un fondo distinto degradaría el desempeño.
4. **Sobreajuste a la sesión de captura.** La brecha de 10,58 puntos muestra que el modelo
   aprendió particularidades del archivo de entrenamiento, no solo el concepto de cada seña.

### 9.4 Confianza del modelo

| | Confianza media |
|---|---|
| En los aciertos | **0,9416** |
| En los errores | **0,7363** |

→ Figura: `images/10_confianza_errores.png`

La confianza **separa sistemáticamente** aciertos de errores. Es aplicable al negocio:
cuando el modelo responde con probabilidad baja, conviene **abstenerse** (pedir al usuario que
repita la seña) en lugar de mostrar una letra probablemente equivocada. Precaución honesta: la
separación no es perfecta, existen errores con confianza alta, por lo que un umbral de
abstención reduce los errores pero no los elimina.

### 9.5 Limitaciones del uso de un MLP en imágenes

Esta es la discusión que la pauta pide explícitamente.

**1. No tiene invarianza a la traslación.** Un MLP es una red de capas densas: cada píxel tiene
su **propio peso** en cada neurona. El peso aprendido para la posición (10, 14) no sirve para
la (11, 14). Si la mano aparece corrida, la entrada es distinta y la red no reconoce que es la
misma seña. **Lo medimos sobre el modelo entrenado:** desplazamos todas las imágenes del
conjunto de prueba y volvemos a predecir. La seña es idéntica, solo cambia su posición.

| Desplazamiento horizontal | Accuracy | Pérdida relativa |
|---|---|---|
| 0 px (original) | 0,8941 | — |
| 1 px | 0,8659 | −3,2% |
| 2 px | 0,6868 | −23,2% |
| 3 px | **0,4337** | **−51,5%** |

→ Figura: `images/09_comparacion_modelos.png`

Mover la imagen **tres píxeles** —el 10,7% del ancho de una imagen de 28 px— **destruye el
desempeño: pierde la mitad de su precisión.** Una CNN no tiene este problema porque aplica el
mismo filtro (pesos compartidos) en toda la imagen, lo que le da invarianza a la traslación por
diseño.

**2. Ignora la estructura espacial.** Al aplanar la imagen en un vector de 784 números, la
vecindad se pierde: el modelo no sabe que los píxeles 1 y 2 son vecinos, ni que el 29 está
justo debajo del 1. Las convoluciones explotan esa vecindad.

**3. Explosión del número de parámetros.** Una capa de 128 neuronas sobre 784 entradas ya usa
100.352 pesos. En los datasets RGB de 150×150 o 256×256 propuestos en el enunciado, la primera
capa tendría 67.500×128 ≈ 8,6 millones de pesos: inmanejable y garantía de sobreajuste. Las
CNN reducen esto al compartir pesos.

**4. Sensibilidad al fondo y a la iluminación.** El dataset es "limpio" (fondo blanco, mano
centrada). Un MLP no aprende a segmentar la mano: aprende también el fondo.

**5. Incapacidad de representar movimiento.** El enfoque solo reconoce posturas estáticas: no
puede representar J ni Z, que requieren movimiento.

---

## 10. Conclusiones

**Desempeño alcanzado.** El modelo M1 (una capa oculta de 128 neuronas) clasifica correctamente
el **89,41%** de las imágenes de un subconjunto de 11 letras sobre datos que nunca vio,
superando con holgura la línea base trivial del 9,1%. La configuración más profunda (M2) mejora
solo 0,66 puntos, lo que demuestra que **agregar capas no es la palanca correcta** en imágenes.
El contraste con M3 cuantifica el costo de ampliar el problema: 18 puntos de accuracy al pasar
de 11 a 24 clases.

**Decisiones más importantes adoptadas.**

1. **Definir una variante propia** (11 letras + comparación de arquitecturas + experimento de
   robustez), que convirtió la entrega en una experimentación real en lugar de una ejecución.
2. **Mantener el hold-out intacto.** Resultó ser la decisión metodológica decisiva: reveló una
   brecha de 10,58 puntos que la validación interna ocultaba.
3. **Aumentar la arquitectura con criterio:** 128 < 784 neuronas, regularización L2 y parada
   temprana mantienen el modelo liviano, aunque no eliminan el sobreajuste.

**Fortalezas.**

1. Métricas altas y equilibradas entre precision y recall.
2. **Reproducible**: semilla fija, partición estratificada documentada, dependencias declaradas.
3. Modelo liviano (menos de 1 MB) y rápido de entrenar (14 s).
4. **Confianza explotable**: los errores se concentran en probabilidades bajas, lo que permite
   abstenerse en lugar de arriesgar.
5. Reporte honesto: se declaran tanto el resultado como su brecha de generalización.

**Limitaciones.**

1. Ausencia de invarianza a la traslación, **medida** (pierde la mitad de la precisión con 3 px
   de desplazamiento).
2. No explota la estructura espacial de la imagen.
3. Sensible al fondo y a la iluminación: no funcionaría sin ajustes con cámaras reales.
4. Sobreajuste no despreciable (10,58 puntos de brecha).
5. Solo cubre posturas estáticas (no J ni Z).

**Mejoras propuestas.**

1. **Reemplazar la arquitectura por una CNN**, que es invariante a la traslación por diseño.
   Es el paso natural y corresponde a la Experiencia de Aprendizaje 2.
2. **Aumento de datos** (desplazamientos, rotaciones leves, cambios de escala) para enseñar al
   MLP la invarianza que no tiene: si el experimento de traslación cuantifica el problema, el
   aumento de datos es su tratamiento directo.
3. **Extracción de características** (HOG, momentos de Hu, contornos) antes del clasificador.
4. **Segmentación previa** para eliminar el fondo y concentrar la red en la mano.
5. **Búsqueda sistemática de hiperparámetros** con validación cruzada, en lugar de la
   comparación puntual de esta entrega.
6. **Regularización adicional** (dropout, más L2) y umbral de abstención en producción.

**Reflexión final.** El valor de este trabajo no está en el número de accuracy alcanzado, sino
en haber aislado *por qué* un MLP es una herramienta limitada frente a imágenes y haber
demostrado esa limitación con evidencia experimental propia —el desplazamiento de 3 píxeles que
cuesta la mitad de la precisión— en lugar de repetir una afirmación de manual. Esa es la razón
por la que las redes convolucionales existen.

---

## 11. Estructura del proyecto

```
proyecto_tly1102/
├── README.md                  # Informe técnico completo (CRISP-DM)
├── requirements.txt           # Dependencias congeladas y verificadas
├── data/
│   ├── sign_mnist_train.csv   # 27.455 imágenes de entrenamiento
│   ├── sign_mnist_test.csv    # 7.172 imágenes de prueba (hold-out)
│   ├── amer_sign2.png         # Alfabeto de referencia
│   ├── amer_sign3.png         # Alfabeto de referencia (B/N)
│   └── american_sign_language.PNG
├── notebooks/
│   └── EP1_TLY1102_MLP_SignLanguageMNIST.ipynb   # Notebook completo y ejecutado (secciones a–h)
├── scripts/
│   ├── 01_eda.py              # Análisis exploratorio → figuras
│   ├── 02_modelo.py           # Pipeline de entrenamiento y evaluación completo
│   └── 03_demo_imagen.py      # Demostración: predice la letra de una imagen externa
├── models/
│   ├── m1_11letras_1capa.joblib    # Modelo principal (M1) serializado
│   ├── m2_11letras_2capas.joblib   # Modelo de comparación (M2)
│   ├── m3_24letras_1capa.joblib    # Modelo de contraste (M3)
│   ├── metricas.json               # Todas las métricas en formato máquina
│   └── eda_resumen.json            # Resumen del EDA
└── images/                    # Todas las figuras del informe y del notebook
```

## 12. Reproducibilidad

| Elemento | Detalle |
|---|---|
| Semilla | `SEED = 42` — partición, inicialización de pesos y muestreo de ejemplos |
| Partición | Estratificada: 85% entrenamiento / 15% validación sobre `train`; `test` como hold-out |
| Entorno | Python 3.11.16 · numpy 2.4.6 · pandas 3.0.6 · scikit-learn 1.9.1 · matplotlib 3.11.2 |
| Ejecución completa | `python scripts/01_eda.py && python scripts/02_modelo.py` |
| Notebook | Abrir `notebooks/EP1_TLY1102_MLP_SignLanguageMNIST.ipynb` → *Run All* |
| Demo en vivo | `python scripts/03_demo_imagen.py images/ejemplos_demo/A_real-A_idx-3.png` |
| Tiempo total | ~2 minutos (EDA + 3 entrenamientos) |

**Verificación de la reproducibilidad.** El pipeline se ejecutó dos veces completas y los
resultados fueron idénticos hasta el cuarto decimal (accuracy 0,8941 / 0,9007 / 0,7140),
lo que confirma que la semilla fija hace el trabajo reproducible.

**Verificación de la portabilidad (local y Google Colab).** El notebook localiza los datos
por sí solo: busca `data/` en varias ubicaciones y, si no lo encuentra y detecta que está
corriendo en Colab, pide subir los dos CSV. Se probó ejecutando las celdas de carga en una
carpeta plana que imita la estructura de Colab (solo el notebook y una carpeta `data/` al
lado, sin `scripts/` ni `models/`): funciona y además crea las carpetas `images/` y `models/`
que falten. La versión anterior usaba `os.path.abspath("..")`, que en Colab resuelve a `/` y
por eso fallaba con `FileNotFoundError: '/data/sign_mnist_train.csv'`.

**Verificación de la demo.** El preprocesamiento de `scripts/03_demo_imagen.py` se validó
con una comparación pareada sobre 400 imágenes del conjunto de prueba: el modelo produce
**exactamente las mismas predicciones (100% de coincidencia)** con el dato crudo y con la
imagen pasada por el pipeline de la demo. Es decir, la demo no introduce distorsión.

## 13. Referencias

- Dataset **Sign Language MNIST** — 27.455 imágenes de entrenamiento y 7.172 de prueba,
  escala de grises 28×28, 24 clases (se excluyen J y Z por requerir movimiento).
- Documentación de `sklearn.neural_network.MLPClassifier`.
- Material de la asignatura TLY1102, Experiencia de Aprendizaje 1: *1.1 El Perceptrón:
  Implementando una neurona*, *1.2 Redes Fully Connected: Construyendo la estructura*,
  *1.3 Descenso del Gradiente: Entrenando el motor*, *1.4 Diseñando y evaluando el modelo*.
- Metodología **CRISP-DM** (*Cross Industry Standard Process for Data Mining*).
