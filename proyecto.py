import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Asegúrate de usar las rutas correctas de los archivos
audio_path_1 = 'D:/Ibai-real_1_.mp3'  # Primer archivo
audio_path_2 = 'D:/IbaiIA.mp3'  # Cambia esto a la ruta correcta del segundo archivo

# Cargar el primer archivo MP3
y1, sr1 = librosa.load(audio_path_1, sr=None)
# Cargar el segundo archivo MP3
y2, sr2 = librosa.load(audio_path_2, sr=None)

# Asegurarse de que ambas frecuencias de muestreo sean iguales
if sr1 != sr2:
    raise ValueError("Los archivos de audio tienen diferentes frecuencias de muestreo.")

# Espectrograma en escala mel del primer archivo
S_mel_1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
S_mel_db_1 = librosa.amplitude_to_db(S_mel_1, ref=np.max)

# Espectrograma en escala mel del segundo archivo
S_mel_2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)
S_mel_db_2 = librosa.amplitude_to_db(S_mel_2, ref=np.max)

# Comparación visual de las diferencias entre los espectrogramas
difference = S_mel_db_1 - S_mel_db_2

# Crear figura con tres subgráficos: dos originales y la diferencia
plt.figure(figsize=(14, 10))

# Primer espectrograma: Archivo 1
plt.subplot(3, 1, 1)
librosa.display.specshow(S_mel_db_1, sr=sr1, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma en escala mel - Audio 1')

# Segundo espectrograma: Archivo 2
plt.subplot(3, 1, 2)
librosa.display.specshow(S_mel_db_2, sr=sr2, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma en escala mel - Audio 2')

# Diferencia entre los espectrogramas
plt.subplot(3, 1, 3)
librosa.display.specshow(difference, sr=sr1, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Diferencias entre los espectrogramas')

plt.tight_layout()
plt.show()

# Comparación numérica
mean_difference = np.mean(np.abs(difference))
print(f'Diferencia media entre los espectrogramas: {mean_difference:.2f} dB')