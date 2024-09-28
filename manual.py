import soundfile as sf
import os

# Ruta al archivo de audio con raw string
file_path = r"C:\Codes\xtts-api-server\output\d8a53b84-f18d-4f5f-bb65-f3f38484a332.wav"

# Verificar si el archivo existe
if os.path.exists(file_path):
    try:
        # Leer el archivo de audio
        info = sf.info(file_path)
        
        # Mostrar la información del archivo de audio
        print(f"Formato: {info.format}")
        print(f"Canales: {info.channels}")
        print(f"Frecuencia de muestreo: {info.samplerate} Hz")
        print(f"Tipo de Subformato: {info.subtype}")
    
    except RuntimeError as e:
        print(f"Error al leer el archivo de audio: {e}")
else:
    print(f"El archivo no existe: {file_path}")
