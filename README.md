# Identificador de Rostros - Modelo (Equipo propio)

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
```

Entrenar es pesado (requiere GPU): usa el notebook en `colab/` si no tienes una local.

Si Colab se desconecta a medio entrenamiento, el notebook monta Google Drive y
guarda ahí los rostros preprocesados y los checkpoints. Vuelve a correr el
notebook desde la celda de Drive y usa `python src/entrenar.py --resume` para
continuar justo donde se quedó en vez de reiniciar desde la fase 1.
