import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import BytesIO

def convertir_y_guardar_espectrograma(archivo_audio, nombre_salida, nombre_audio):
    # Cargar el audio
    y, sr = librosa.load(archivo_audio, sr=None)

    # Generar el espectrograma Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # Convertir el espectrograma a escala logarítmica
    S_db = librosa.power_to_db(S, ref=np.max)

    # Crear una imagen del espectrograma usando matplotlib
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

    # Añadir el título (nombre del archivo de audio) en la parte superior
    #plt.title(f"Espectrograma: {nombre_audio}", fontsize=12)

    # Guardar la imagen del espectrograma en la carpeta de salida
    ruta_imagen = os.path.join(carpeta_salida_espectrogramas, nombre_salida)
    plt.tight_layout()
    plt.savefig(ruta_imagen, format='png')
    plt.close()

    return ruta_imagen

carpeta_salida_espectrogramas = '/Proyecto SIC/espectrogramas_IA'

# Crear la carpeta para guardar los espectrogramas si no existe
if not os.path.exists(carpeta_salida_espectrogramas):
    os.makedirs(carpeta_salida_espectrogramas)

# Crear listas vacías para almacenar la información de los archivos
nombres_archivos = []
duraciones = []
frecuencias_muestreo = []
rutas_imagenes_espectrograma = []
etiquetas = []

# Obtener la lista de archivos en la carpeta
archivos_audio = [f for f in os.listdir('Proyecto SIC/Voces IA') if f.endswith('.mp3') or f.endswith('.wav')]

# Recorrer cada archivo de audio y procesarlo
for archivo in archivos_audio:
    ruta_archivo = os.path.join('Proyecto SIC\Voces IA', archivo)

    # Cargar el archivo de audio usando librosa
    y, sr = librosa.load(ruta_archivo, sr=None)

    # Guardar la información básica del archivo
    nombres_archivos.append(archivo)
    duraciones.append(librosa.get_duration(y=y, sr=sr))
    frecuencias_muestreo.append(sr)

    # Generar y guardar el espectrograma como una imagen PNG con el nombre del archivo de audio
    print(f"Procesando archivo: {archivo}")
    nombre_salida_espectrograma = archivo.replace('.mp3', '.png').replace('.wav', '.png')
    ruta_imagen = convertir_y_guardar_espectrograma(ruta_archivo, nombre_salida_espectrograma, archivo)
    rutas_imagenes_espectrograma.append(ruta_imagen)

    # Asignar una etiqueta basada en el nombre del archivo (ajusta esta lógica según el dataset)
    etiqueta = 1 if 'real' in archivo.lower() else 0
    etiquetas.append(etiqueta)

# Crear un DataFrame con la información de los archivos y las rutas de los espectrogramas
df = pd.DataFrame({
    'Nombre del archivo': nombres_archivos,
    'Duración (segundos)': duraciones,
    'Frecuencia de muestreo': frecuencias_muestreo,
    'Ruta espectrograma': rutas_imagenes_espectrograma,
    'Etiqueta': etiquetas
})

df.head()

# Guardar el DataFrame como archivo CSV en la ruta correcta
ruta_csv = os.path.join(carpeta_salida_espectrogramas, 'dataset_audio_IA.csv')
df.to_csv(ruta_csv, index=False)

# Mostrar el DataFrame
df.head()

# Supongamos que `espectrograma_base64` es la cadena base64 de una de tus imágenes en el DataFrame
espectrograma_base64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAIyUlEQVR4nO3WMQEAIAzAMMC/5+ECjiYKenbPzCwAADLO7wAAAN4ygAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIuJnkHvKensmIAAAAASUVORK5CYII="  # Por ejemplo, mostrar la primera imagen

# Decodificar la imagen base64
imagen_decodificada = base64.b64decode(espectrograma_base64)

# Convertir la imagen en un objeto de imagen que puede ser mostrado
imagen = Image.open(BytesIO(imagen_decodificada))

# Mostrar la imagen
plt.imshow(imagen)
plt.axis('off')  # Ocultar los ejes
plt.show()