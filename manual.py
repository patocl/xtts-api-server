import soundfile as sf

filename = 'speakers/male_fixed.wav'

# Intenta cargar el archivo para verificar que el encabezado es correcto
try:
    data, samplerate = sf.read(filename)
    print(f"Archivo cargado correctamente. Muestreo: {samplerate}, Datos: {data.shape}")
except Exception as e:
    print(f"Error al cargar el archivo WAV: {e}")
