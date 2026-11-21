# Guía para el equipo: entrenar el modelo en Colab

Esta guía explica paso a paso cómo correr el notebook de entrenamiento, cómo
funciona el guardado de checkpoints, y qué hacer si Colab se desconecta a
medio proceso. No necesitas entender todo el código, solo seguir el orden.

## 0. Requisitos antes de empezar

- Cuenta de Hugging Face agregada como **miembro** (rol "read" o superior) de
  la organización `escom-identificador-rostros`. Sin esto no puedes descargar
  el dataset. Si no te han agregado, avísale a Gadiel.
- Cuenta de Google (para Colab y para que se monte tu Google Drive).
- Una foto tuya o de cualquier persona del dataset, si quieres probar el
  modelo al final (paso 7).

## 1. Abrir el notebook

Entra a este link:

**https://colab.research.google.com/github/Gad-gg-23/Identificador_Rostros_Modelo/blob/main/colab/Entrenamiento_Modelo.ipynb**

Antes de correr nada: **Entorno de ejecución → Cambiar tipo de entorno de
ejecución → GPU**. Sin GPU, entrenar puede tardar horas en vez de minutos.

## 2. Correr las celdas en orden

El notebook está pensado para correrse de arriba hacia abajo, celda por
celda (Shift+Enter en cada una). No te saltes ninguna.

1. **Clonar + instalar dependencias**: clona este repo dentro de Colab e
   instala `requirements.txt`. Tarda 1-2 minutos.
2. **Login a Hugging Face**: te va a pedir un token. Sácalo de
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (uno de tipo "read" es suficiente) y pégalo cuando te lo pida.
3. **Descargar el dataset**: baja las ~30,000 fotos del dataset privado a
   `data/`. Puede tardar varios minutos; Hugging Face a veces da errores de
   red o "rate limit" (HTTP 429) a mitad de la descarga — si se corta, **solo
   vuelve a correr la misma celda**, retoma donde se quedó en vez de empezar
   de cero.
4. **Montar Google Drive**: te va a pedir autorizar acceso a tu Drive. Esto
   crea una carpeta `ESCOM_ML_Identificador_Rostros/` en tu Drive y hace que
   **todo lo que se guarde de aquí en adelante (rostros procesados y
   checkpoints) viva ahí**, no en la máquina temporal de Colab. Ver sección
   "¿Por qué Google Drive?" más abajo.
5. **Detectar y alinear rostros (MTCNN)**: procesa cada foto cruda y guarda
   el rostro recortado en tu Drive. Es la parte más lenta (puede tardar
   30-60 min con GPU). Si se desconecta a medio camino, vuelve a correr esta
   misma celda: los rostros que ya estén guardados en Drive se saltan
   automáticamente y solo procesa los que faltan.
6. **Entrenar**: dos fases (primero solo la "cabeza" del modelo, luego se
   afina el backbone completo). Esta celda ya incluye `--resume`, así que si
   Colab se desconecta a medio entrenamiento, **solo vuelve a correr esta
   celda** (después de volver a correr la celda 4 de montar Drive) y continúa
   exactamente donde se quedó, sin reiniciar desde el principio. Ver sección
   "¿Qué pasa si Colab se desconecta?" más abajo.
7. **Evaluar**: imprime el accuracy por persona y guarda una matriz de
   confusión (`matriz_confusion.png`) en la carpeta de checkpoints de tu
   Drive.
8. **Descargar el checkpoint entrenado**: descarga `mejor_modelo.pt` a tu
   computadora, para poder compartirlo con el equipo o probarlo localmente.
9. **Probar una foto nueva**: sube cualquier foto y el modelo te dice a qué
   persona cree que corresponde (top-3 con porcentaje de confianza).

## 3. ¿Por qué Google Drive?

La máquina virtual que te da Colab es **temporal**: si se desconecta (por
inactividad, por los límites de tiempo de la cuenta gratuita, o porque se
cierra la pestaña), **todo lo que esté solo en esa máquina se borra**,
incluyendo rostros procesados y checkpoints a medio entrenar.

Por eso la celda 4 monta tu Google Drive y redirige ahí dos carpetas:

- `rostros/` → rostros ya recortados/alineados (para no tener que volver a
  correr MTCNN sobre las 30,000 fotos si se corta la sesión).
- `checkpoints/` → el mejor modelo guardado (`mejor_modelo.pt`) y el estado
  completo de entrenamiento (`ultimo_estado.pt`), que es lo que permite
  reanudar con `--resume`.

Todo esto queda dentro de `Mi unidad/ESCOM_ML_Identificador_Rostros/` en tu
Drive. Pesa unos cuantos GB (sobre todo los rostros), así que asegúrate de
tener espacio libre.

## 4. ¿Qué pasa si Colab se desconecta?

No pasa nada grave, solo sigue estos pasos:

1. Abre el notebook otra vez (o la misma pestaña si Colab te ofrece
   reconectar).
2. Vuelve a correr las celdas 1 a 4 (clonar, login, descargar dataset si
   hiciera falta, montar Drive) — son rápidas o se saltan lo que ya existe.
3. Corre la celda de "Detectar rostros" otra vez si no habías terminado esa
   parte: salta lo ya hecho.
4. Corre la celda de "Entrenar" otra vez: como ya trae `--resume`, retoma
   justo en la fase y época donde se quedó (vas a ver un mensaje como
   `Reanudando desde fase2, epoca 7 (mejor_acc=0.8421)`).

**Importante**: si corres `entrenar.py` manualmente desde una terminal (no
el notebook) sin el flag `--resume`, va a reiniciar desde cero aunque exista
un checkpoint previo. El `--resume` es obligatorio para continuar.

## 5. Compartir el modelo entrenado con el equipo

`checkpoints/mejor_modelo.pt` **no se sube a GitHub** (está en
`.gitignore` a propósito: son pesos de un modelo entrenado con fotos
biométicas del equipo, no debe quedar en un repo público). Para compartirlo:

- Comparte el archivo descargado en el paso 8 (por Drive, WhatsApp, Classroom,
  lo que sea), o
- Comparte directamente el link de la carpeta de tu Google Drive
  (`ESCOM_ML_Identificador_Rostros/checkpoints/`) con quien lo necesite.

## 6. Probar el modelo en una computadora normal (sin GPU)

Una vez que tienes `mejor_modelo.pt`, probar una foto **no necesita GPU**
(a diferencia de entrenar, que sí). Con Python y las dependencias de
`requirements.txt` instaladas:

```bash
python src/inferir.py --imagen foto.jpg --checkpoint ruta/a/mejor_modelo.pt
```

Tarda 1-2 segundos por foto. El checkpoint ya incluye el mapeo de índice a
persona, así que no necesitas nada más del entrenamiento.

## 7. Problemas comunes

- **"HTTP Error 429" / "Rate limited" al descargar el dataset**: normal,
  Hugging Face limita descargas grandes. Vuelve a correr la misma celda.
- **Error de conexión (`WinError`/`ConnectError`) a medio proceso**: lo
  mismo, vuelve a correr la celda; `snapshot_download` retoma donde se quedó.
- **"No se encontraron fotos en data" / "No se encontraron rostros en
  rostros"**: te saltaste una celda anterior (descargar dataset o
  preprocesar rostros), o estás corriendo el script desde la carpeta
  equivocada. Corre las celdas en orden desde el principio.
- **Se te acabó la cuota de GPU gratis de Colab**: espera unas horas a que se
  renueve, o usa otra cuenta de Google. Mientras tanto no pierdes nada: todo
  lo avanzado ya está en tu Drive.
