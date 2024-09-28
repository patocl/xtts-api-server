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
response = requests.post(url, json=params, stream=True)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Crear un archivo para guardar la salida u-law
    output_filename = "output_ulaw.wav"

    # Preparar un buffer para almacenar los datos de audio u-law
    ulaw_data = b""

    # Leer la respuesta en streaming
    for chunk in response.iter_lines():
        if chunk:
            # Decodificar cada bloque Base64 a bytes
            ulaw_data += base64.b64decode(chunk)

    # Escribir los datos u-law en un archivo
    with io.BytesIO(ulaw_data) as ulaw_io:
        data, samplerate = sf.read(ulaw_io, dtype='int16')
        sf.write(output_filename, data, 8000, subtype='ULAW')

    print(f"Audio saved as {output_filename}")
else:
    print(f"Failed to retrieve audio: {response.status_code}")
