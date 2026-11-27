# Identificador de Rostros - Modelo (Equipo propio)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gad-gg-23/Identificador_Rostros_Modelo/blob/main/colab/Entrenamiento_Modelo.ipynb)
[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Gad-gg-23/Identificador_Rostros_Modelo/blob/main/kaggle/Entrenamiento_Modelo_Kaggle.ipynb)

Repo de **código del modelo** para la práctica de identificación de compañeros por
rostro (ESCOM ML). Los datos (fotos) viven en un repo aparte, solo de datos:
https://github.com/Gad-gg-23/Dataset-Identificador-Rostros-ESCOM

## Arquitectura

En vez de entrenar una red convolucional desde cero (con ~1000 fotos por persona
y ~28 personas, los resultados serían mediocres), usamos **transfer learning**:

1. **Backbone**: `InceptionResnetV1` preentrenado en VGGFace2 ([`facenet-pytorch`](https://github.com/timesler/facenet-pytorch)),
   ya experto en reconocimiento facial — solo lo afinamos para nuestras personas.
2. **Cabeza ArcFace** (margen angular aditivo) en vez de softmax normal: es la
   técnica estándar en reconocimiento facial de producción, separa mejor las
   identidades en el espacio de embeddings.
3. **Entrenamiento en 2 fases**: primero solo la cabeza (backbone congelado),
   luego se descongela el último bloque del backbone con LR bajo (fine-tuning).
4. **Split train/val por *set* completo**, no por foto suelta: cada set es una
   sesión con ropa/fondo distintos, así que separar por set mide si el modelo
   aprendió rasgos faciales reales y no memorizó ropa o fondo — justo el
   objetivo que marca el dataset.
5. **Augmentación con oclusión simulada** (`RandomErasing`, recorte aleatorio de
   regiones) para ganar algo de robustez a lentes/gorras/etc. sin necesitar
   miles de fotos disfrazadas — el disfraz en el dataset es minoritario a propósito.

## Estructura

```
Identificador_Rostros_Modelo/
├── config.yaml              # hiperparámetros
├── descargar_datos.py       # baja el dataset desde Hugging Face a data/
├── requirements.txt
├── src/
│   ├── preprocesar_rostros.py  # detecta/alinea rostros con MTCNN -> rostros/
│   ├── dataset.py              # indice, split por set, Dataset + augmentacion
│   ├── modelo.py                # backbone + cabeza ArcFace
│   ├── entrenar.py              # entrenamiento en 2 fases + early stopping
│   ├── evaluar.py               # accuracy, reporte por persona, matriz de confusion
│   └── inferir.py               # predice la identidad de una foto nueva
├── colab/
│   └── Entrenamiento_Modelo.ipynb  # entrenamiento con GPU gratis de Colab
├── kaggle/
│   └── Entrenamiento_Modelo_Kaggle.ipynb  # mismo entrenamiento con GPU gratis de Kaggle
└── checkpoints/              # NO va en git: mejor_modelo.pt, logs, graficas
```

## Uso

```bash
pip install -r requirements.txt
hf auth login                              # cuenta miembro de la organizacion

python descargar_datos.py                  # -> data/persona/set_xx/archivo.jpg
python src/preprocesar_rostros.py          # -> rostros/persona/set_xx/archivo.jpg
python src/entrenar.py                     # -> checkpoints/mejor_modelo.pt
python src/evaluar.py                      # accuracy + matriz de confusion
python src/inferir.py --imagen foto.jpg    # predice una foto nueva
python src/app.py                          # interfaz grafica (sube foto o usa camara)
```

Entrenar es pesado (requiere GPU): usa el notebook en `colab/` o en `kaggle/` si no
tienes una local (botones al inicio de este README).

En Kaggle, antes de correr: activa **GPU** e **Internet** en *Settings* y agrega tu token
de Hugging Face como secreto `HF_TOKEN` (*Add-ons > Secrets*). Usa *Save Version >
Save & Run All* para que corra en segundo plano; los rostros y checkpoints quedan en el
*Output* del notebook. Para reanudar una corrida cortada, agrega ese Output como *Input*.

`python src/app.py` abre una interfaz web local (Gradio) en
http://127.0.0.1:7860 para subir una foto o usar la camara y ver la
prediccion con barras de confianza, sin tener que usar la linea de comandos.
No requiere GPU ni reentrenar nada, solo el checkpoint ya entrenado.

Si Colab se desconecta a medio entrenamiento, el notebook monta Google Drive y
guarda ahí los rostros preprocesados y los checkpoints. Vuelve a correr el
notebook desde la celda de Drive y usa `python src/entrenar.py --resume` para
continuar justo donde se quedó en vez de reiniciar desde la fase 1.
