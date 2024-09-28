import requests
import base64
import soundfile as sf
import io

# URL del endpoint (ajusta la URL si es necesario)
url = "http://127.0.0.1:8002/tts_to_ulaw"

# Parámetros para la solicitud
params = {
    'text': 'la vaca Lola tiene cabeza y tiene cola y hace Muuu!',
    'speaker_wav': 'male_fixed.wav',
    'language': 'es'
}

# Realizar la solicitud POST al endpoint
response = requests.post(url, json=params)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    try:
        # Obtener la respuesta en formato JSON
        response_json = response.json()

        # Extraer los datos en base64 del campo "audio_base64"
        ulaw_base64 = response_json.get('audio_base64', None)

        if ulaw_base64:
            # Decodificar el audio base64 a bytes
            ulaw_data = base64.b64decode(ulaw_base64)

            # Crear un buffer de memoria para trabajar con el archivo
            with io.BytesIO(ulaw_data) as ulaw_io:
                # Leer el archivo de audio
                data, samplerate = sf.read(ulaw_io, dtype='int16')

                # Validar si el archivo tiene la tasa de muestreo de 8000 Hz
                if samplerate == 8000:
                    print("Tasa de muestreo validada: 8000 Hz.")
                else:
                    print(f"Tasa de muestreo incorrecta: {samplerate} Hz. Se esperaba 8000 Hz.")

                # Guardar el archivo como salida para su análisis
                output_filename = "output_ulaw.wav"
                sf.write(output_filename, data, samplerate, subtype='ULAW')
                print(f"Archivo guardado como {output_filename}")

                # Mostrar información sobre el archivo guardado
                info = sf.info(output_filename)
                print(f"Formato: {info.format}")
                print(f"Canales: {info.channels}")
                print(f"Frecuencia de muestreo: {info.samplerate} Hz")
                print(f"Tipo de Subformato: {info.subtype}")

                # Validar que sea u-law y que tenga 1 canal
                if info.subtype == 'ULAW' and info.channels == 1:
                    print("El archivo está correctamente en formato u-law con 1 canal.")
                else:
                    print(f"Formato o canales incorrectos. Subformato: {info.subtype}, Canales: {info.channels}")

        else:
            print("No se encontró el campo 'audio_base64' en la respuesta.")
    except Exception as e:
        print(f"Error al procesar la respuesta: {e}")
else:
    print(f"Error al obtener el audio: {response.status_code}")
