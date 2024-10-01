# Bibliotecas estándar
import os
import io
import time
import base64
import shutil
from pathlib import Path
from uuid import uuid4
from argparse import ArgumentParser
import logging

# Bibliotecas de terceros
import soundfile as sf
from pydub import AudioSegment
from starlette.responses import StreamingResponse
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Importaciones locales/personalizadas
from TTS.api import TTS
from xtts_api_server.tts_funcs import (
    TTSWrapper,
    supported_languages,
    InvalidSettingsError
)
from xtts_api_server.RealtimeTTS import TextToAudioStream, CoquiEngine
from xtts_api_server.modeldownloader import (
    check_stream2sentence_version,
    install_deepspeed_based_on_python_version
)

logger = logging.getLogger(__name__)
# Constants (environment variables or default values)
DEVICE = os.getenv('DEVICE', "cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
MODEL_FOLDER = os.getenv('MODEL', 'models')
BASE_HOST = os.getenv('BASE_URL', '127.0.0.1:8020')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0.2")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
USE_CACHE = os.getenv("USE_CACHE") == 'true'

STREAM_MODE = os.getenv("STREAM_MODE") == 'true'
STREAM_MODE_IMPROVE = os.getenv("STREAM_MODE_IMPROVE") == 'true'
STREAM_PLAY_SYNC = os.getenv("STREAM_PLAY_SYNC") == 'true'

if DEEPSPEED:
    try:
        install_deepspeed_based_on_python_version()
    except Exception as e:
        logger.exception("Error installing DeepSpeed: %s", e)
        raise

# FastAPI instance and TTSWrapper initialization
app = FastAPI()

try:
    XTTS = TTSWrapper(
        OUTPUT_FOLDER, SPEAKER_FOLDER, MODEL_FOLDER, LOWVRAM_MODE,
        MODEL_SOURCE, MODEL_VERSION, DEVICE, DEEPSPEED, USE_CACHE
    )
except Exception as e:
    logger.exception("Error initializing TTSWrapper: %s", e)
    raise

# Model version checking
try:
    XTTS.model_version = XTTS.check_model_version_old_format(MODEL_VERSION)
    MODEL_VERSION = XTTS.model_version
except Exception as e:
    logger.exception("Error checking model version: %s", e)
    raise

if MODEL_SOURCE == "api" or MODEL_VERSION == "main":
    version_string = "latest"
else:
    version_string = MODEL_VERSION

# Load model
try:
    if STREAM_MODE or STREAM_MODE_IMPROVE:
        check_stream2sentence_version()
        logger.warning("'Streaming Mode' has certain limitations. More details here: https://github.com/daswer123/xtts-api-server#about-streaming-mode")

        if STREAM_MODE_IMPROVE:
            logger.info("You have launched an improved version of streaming, featuring better tokenization and more context for complex languages like Chinese.")

        model_path = XTTS.model_folder
        engine = CoquiEngine(
            specific_model=MODEL_VERSION,
            use_deepspeed=DEEPSPEED,
            local_models_path=str(model_path)
        )
        stream = TextToAudioStream(engine)
    else:
        logger.info(f"Model '{version_string}' is loading. Please wait.")
        XTTS.load_model()
except Exception as e:
    logger.exception("Error loading model: %s", e)
    raise

if USE_CACHE:
    logger.info("Cache is enabled, allowing the reuse of previously generated results.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def play_stream(stream, language):
    """
    Play the audio stream with the specified settings.

    Parameters:
    - stream: The audio stream object to be played.
    - language: The language code to be used in the stream.
    """
    play_args = {
        'minimum_sentence_length': 2,
        'minimum_first_fragment_length': 2,
        'tokenizer': "stanza",
        'language': language,
        'context_size': 2
    }

    try:
        if STREAM_MODE_IMPROVE:
            if STREAM_PLAY_SYNC:
                stream.play(**play_args)
            else:
                stream.play_async(**play_args)
        else:
            if STREAM_PLAY_SYNC:
                stream.play()
            else:
                stream.play_async()
    except Exception as e:
        logger.exception("Error playing stream: %s", e)
        raise

# Input models for the requests
class OutputFolderRequest(BaseModel):
    output_folder: str

class SpeakerFolderRequest(BaseModel):
    speaker_folder: str

class ModelNameRequest(BaseModel):
    model_name: str

class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str

class SynthesisFileRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str
    file_name_or_path: str

class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str

# API Endpoints with detailed documentation

@app.get("/speakers_list")
def get_speakers():
    """
    Get the list of available speakers.

    ### Returns:
    - A list of available speakers, either preloaded or user-provided.

    ### Example:
    - `GET /speakers_list` will return:
    ```json
    [
        {"id": "speaker1", "name": "Speaker 1"},
        {"id": "speaker2", "name": "Speaker 2"}
    ]
    ```
    """
    try:
        return XTTS.get_speakers()
    except Exception as e:
        logger.exception("Error fetching speaker list: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/languages")
def get_languages():
    """
    Get the list of supported languages.

    ### Returns:
    - A JSON object containing the supported languages.

    ### Example:
    - `GET /languages` will return:
    ```json
    {
        "languages": ["en", "es", "fr"]
    }
    ```
    """
    try:
        return {"languages": XTTS.list_languages()}
    except Exception as e:
        logger.exception("Error fetching languages: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/get_folders")
def get_folders():
    """
    Retrieve the currently configured folders.

    ### Returns:
    - A JSON object with the paths of the speaker folder, output folder, and model folder.

    ### Example:
    - `GET /get_folders` will return:
    ```json
    {
        "speaker_folder": "/path/to/speakers",
        "output_folder": "/path/to/output",
        "model_folder": "/path/to/models"
    }
    ```
    """
    try:
        return {
            "speaker_folder": XTTS.speaker_folder,
            "output_folder": XTTS.output_folder,
            "model_folder": XTTS.model_folder
        }
    except Exception as e:
        logger.exception("Error fetching folder paths: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/get_models_list")
def get_models_list():
    """
    Retrieve the list of available models.

    ### Returns:
    - A list of TTS models that are currently available for use.

    ### Example:
    - `GET /get_models_list` will return:
    ```json
    [
        {"id": "model1", "name": "TTS Model 1"},
        {"id": "model2", "name": "TTS Model 2"}
    ]
    ```
    """
    try:
        return XTTS.get_models_list()
    except Exception as e:
        logger.exception("Error fetching models list: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/get_tts_settings")
def get_tts_settings():
    """
    Retrieve the current TTS settings.

    ### Returns:
    - A JSON object with the current TTS settings, including stream chunk size and other parameters.

    ### Example:
    - `GET /get_tts_settings` will return:
    ```json
    {
        "stream_chunk_size": 1024,
        "temperature": 0.7,
        "speed": 1.0,
        ...
    }
    ```
    """
    try:
        return {**XTTS.tts_settings, "stream_chunk_size": XTTS.stream_chunk_size}
    except Exception as e:
        logger.exception("Error fetching TTS settings: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/set_output")
def set_output(output_req: OutputFolderRequest):
    """
    Set the output folder for the generated audio files.

    ### Parameters:
    - `output_folder` (str): The path to the folder where the generated audio files will be stored.

    ### Returns:
    - A JSON object with a confirmation message indicating that the output folder has been successfully set.
    
    ### Example:
    - Request:
    ```json
    {
        "output_folder": "/new/output/folder"
    }
    ```
    - Response:
    ```json
    {
        "message": "Output folder set to /new/output/folder"
    }
    ```

    ### Error Handling:
    - If the folder path is invalid, the function will return a `400 Bad Request` with an error message.
    - If an unexpected error occurs, the function will return a `500 Internal Server Error`.
    """
    try:
        XTTS.set_out_folder(output_req.output_folder)
        return {"message": f"Output folder set to {output_req.output_folder}"}
    except ValueError as e:
        logger.error("Error setting output folder: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error setting output folder: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/set_speaker_folder")
def set_speaker_folder(speaker_req: SpeakerFolderRequest):
    """
    Set the folder where speaker audio files are stored.

    ### Parameters:
    - `speaker_folder` (str): The path to the folder where speaker audio files are stored.

    ### Returns:
    - A JSON object with a confirmation message indicating that the speaker folder has been successfully set.

    ### Example:
    - Request:
    ```json
    {
        "speaker_folder": "/new/speaker/folder"
    }
    ```
    - Response:
    ```json
    {
        "message": "Speaker folder set to /new/speaker/folder"
    }
    ```

    ### Error Handling:
    - If the folder path is invalid, the function will return a `400 Bad Request` with an error message.
    - If an unexpected error occurs, the function will return a `500 Internal Server Error`.
    """
    try:
        XTTS.set_speaker_folder(speaker_req.speaker_folder)
        return {"message": f"Speaker folder set to {speaker_req.speaker_folder}"}
    except ValueError as e:
        logger.error("Error setting speaker folder: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error setting speaker folder: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/switch_model")
def switch_model(model_req: ModelNameRequest):
    """
    Switch the active TTS model to a different one.

    ### Parameters:
    - `model_name` (str): The name of the model to switch to. This should be a valid model available in the system.

    ### Returns:
    - A JSON object with a confirmation message indicating that the model has been successfully switched.

    ### Example:
    - Request:
    ```json
    {
        "model_name": "new_model"
    }
    ```
    - Response:
    ```json
    {
        "message": "Model switched to new_model"
    }
    ```

    ### Error Handling:
    - If the model name is invalid or unsupported, the function will return a `400 Bad Request` with an error message.
    - If an unexpected error occurs, the function will return a `500 Internal Server Error`.

    ### Notes:
    - The model switch is performed using the internal TTS engine, and the new model should be available in the configured model folder or via the API source.
    """
    try:
        XTTS.switch_model(model_req.model_name)
        return {"message": f"Model switched to {model_req.model_name}"}
    except InvalidSettingsError as e:
        logger.error("Error switching model: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error switching model: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/set_tts_settings")
def set_tts_settings_endpoint(tts_settings_req: TTSSettingsRequest):
    """
    Set the Text-to-Speech (TTS) model's settings, which control various aspects of the TTS engine's behavior.

    ### Parameters:
    - `stream_chunk_size` (int): Size of the stream chunks (buffer size).
    - `temperature` (float): Controls the randomness of the TTS model's output (0.0 is deterministic).
    - `speed` (float): Adjusts the speaking speed of the generated speech.
    - `length_penalty` (float): Penalizes longer sentences, encouraging shorter outputs.
    - `repetition_penalty` (float): Penalizes repetitive sentences in the output.
    - `top_p` (float): Controls the cumulative probability threshold for token selection during text generation.
    - `top_k` (int): Limits the number of top tokens considered for generation (smaller values make generation more deterministic).
    - `enable_text_splitting` (bool): Whether to enable splitting long texts into smaller, more manageable chunks for processing.

    ### Returns:
    - A JSON object with a confirmation message indicating that the TTS settings have been successfully applied.

    ### Example:
    - Request:
    ```json
    {
        "stream_chunk_size": 1024,
        "temperature": 0.7,
        "speed": 1.0,
        "length_penalty": 1.0,
        "repetition_penalty": 1.2,
        "top_p": 0.9,
        "top_k": 50,
        "enable_text_splitting": true
    }
    ```
    - Response:
    ```json
    {
        "message": "Settings successfully applied"
    }
    ```

    ### Error Handling:
    - If any of the settings are invalid or not supported, the function will return a `400 Bad Request` with an error message.
    - If an unexpected error occurs, the function will return a `500 Internal Server Error`.

    ### Notes:
    - The settings apply immediately to the TTS engine and will affect subsequent speech synthesis requests.
    - Ensure the values provided are within valid ranges to prevent errors or unexpected behavior.
    """
    try:
        XTTS.set_tts_settings(**tts_settings_req.dict())
        return {"message": "Settings successfully applied"}
    except InvalidSettingsError as e:
        logger.error("Error in TTS settings: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error applying TTS settings: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

# TTS streaming
@app.get('/tts_stream')
async def tts_stream(
    request: Request,
    text: str = Query(...),
    speaker_wav: str = Query(...),
    language: str = Query(...)
):
    """
    Generate an audio stream from the provided text using the TTS model, allowing real-time playback.

    ### Parameters:
    - `text` (str): The text to be synthesized into speech.
    - `speaker_wav` (str): The path or name of the speaker's audio file to be used for voice cloning.
    - `language` (str): The language code for the synthesis (e.g., 'en' for English, 'es' for Spanish).

    ### Returns:
    - A streaming audio response (`audio/x-wav`) that streams the generated speech in real-time.

    ### Example:
    - Request:
    ```
    GET /tts_stream?text=Hello%20world&speaker_wav=speaker1.wav&language=en
    ```

    - Response:
    The response is a streaming audio file in WAV format, played in real-time.

    ### Error Handling:
    - If the TTS model is not local, the function will return a `400 Bad Request`, as HTTP streaming is only supported for local models.
    - If the language code is unsupported or misspelled, the function will return a `400 Bad Request`.
    - If an unexpected error occurs during the streaming process, the function will return a `500 Internal Server Error`.

    ### Notes:
    - This endpoint supports real-time streaming, which is optimal for applications requiring immediate playback of generated speech.
    - Ensure that the speaker file and language code are correctly set to avoid errors during the streaming process.
    """
    try:
        # Validate that the TTS model is local
        if XTTS.model_source != "local":
            raise HTTPException(status_code=400, detail="HTTP Streaming is only supported for local models.")

        # Validate the language code
        if language.lower() not in supported_languages:
            raise HTTPException(status_code=400, detail="Unsupported or misspelled language code.")

        # Generator function for streaming audio chunks
        async def generator():
            try:
                # Process TTS and stream audio chunks
                chunks = XTTS.process_tts_to_file(
                    text=text,
                    speaker_name_or_path=speaker_wav,
                    language=language.lower(),
                    stream=True
                )
                # Write WAV header to the stream
                yield XTTS.get_wav_header()

                # Stream each audio chunk
                async for chunk in chunks:
                    if await request.is_disconnected():
                        break
                    yield chunk
            except Exception as e:
                logger.exception("Error during audio streaming: %s", e)
                raise HTTPException(status_code=500, detail="Internal server error during audio streaming")

        # Return the streaming response
        return StreamingResponse(generator(), media_type='audio/x-wav')

    except HTTPException as e:
        logger.error("Error in TTS stream request: %s", e.detail)
        raise
    except Exception as e:
        logger.exception("Unexpected error during TTS streaming: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Convert text into speech and return the generated audio as a file. Supports both streaming and non-streaming modes.

    ### Parameters:
    - `text` (str): The text to be synthesized into speech.
    - `speaker_wav` (str): The path or name of the speaker's audio file to be used for voice cloning.
    - `language` (str): The language code for the synthesis (e.g., 'en' for English, 'es' for Spanish).

    ### Returns:
    - A `FileResponse` containing the generated speech as a WAV audio file.

    ### Streaming Mode:
    - If `STREAM_MODE` or `STREAM_MODE_IMPROVE` is enabled, the audio will be streamed in real-time.
    - If streaming is disabled, the TTS engine will process the text and return a fully generated audio file.

    ### Example (Non-Streaming):
    - Request:
    ```json
    {
        "text": "Hello, world!",
        "speaker_wav": "speaker1.wav",
        "language": "en"
    }
    ```
    - Response:
    A `FileResponse` with the audio content of the synthesized speech.

    ### Example (Streaming):
    - Request:
    ```json
    {
        "text": "Hello, world!",
        "speaker_wav": "speaker1.wav",
        "language": "en"
    }
    ```
    - Response:
    A 1-second silent audio file to initiate the stream.

    ### Error Handling:
    - If the language code is unsupported or misspelled, the function will return a `400 Bad Request`.
    - If an error occurs during the speech synthesis, the function will return a `500 Internal Server Error`.

    ### Notes:
    - The function supports both synchronous and asynchronous playback modes.
    - If the speaker file or language code is invalid, the synthesis process will raise an error.
    """
    # Check if streaming mode is enabled
    if STREAM_MODE or STREAM_MODE_IMPROVE:
        try:
            global stream

            # Validate the language code
            if request.language.lower() not in supported_languages:
                raise HTTPException(status_code=400, detail="Unsupported or misspelled language code.")

            speaker_wav = XTTS.get_speaker_wav(request.speaker_wav)
            language = request.language[0:2]

            # Stop the stream if it's already playing and not in sync mode
            if stream.is_playing() and not STREAM_PLAY_SYNC:
                stream.stop()
                stream = TextToAudioStream(engine)

            # Set the voice and language
            engine.set_voice(speaker_wav)
            engine.language = request.language.lower()

            # Start feeding the stream with the text to generate audio
            stream.feed(request.text)
            play_stream(stream, language)

            # Send 1 second of silence to avoid client errors while the stream starts
            this_dir = Path(__file__).parent.resolve()
            output = this_dir / "RealtimeTTS" / "silence.wav"

            return FileResponse(
                path=output,
                media_type='audio/wav',
                filename="silence.wav"
            )
        except HTTPException as e:
            logger.error("Error in request: %s", e.detail)
            raise
        except Exception as e:
            logger.exception("Error processing TTS to audio in streaming mode: %s", e)
            raise HTTPException(status_code=500, detail="Internal server error")

    else:
        try:
            # Validate the language code
            if request.language.lower() not in supported_languages:
                raise HTTPException(status_code=400, detail="Unsupported or misspelled language code.")

            # Process the text and generate the audio file
            output_file_path = XTTS.process_tts_to_file(
                text=request.text,
                speaker_name_or_path=request.speaker_wav,
                language=request.language.lower(),
                file_name_or_path=f'{str(uuid4())}.wav'
            )

            # If caching is disabled, add a background task to remove the file after use
            if not XTTS.enable_cache_results:
                background_tasks.add_task(os.unlink, output_file_path)

            # Return the generated audio file
            return FileResponse(
                path=output_file_path,
                media_type='audio/wav',
                filename="output.wav"
            )
        except HTTPException as e:
            logger.error("Error in request: %s", e.detail)
            raise
        except Exception as e:
            logger.exception("Error processing TTS to audio: %s", e)
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get('/tts_ulaw')
async def tts_to_ulaw(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Convert text into speech, save the result as a .ulaw file with an 8kHz sample rate,
    and return the generated u-law audio file.
    
    ### Parameters:
    - `text` (str): The text to be synthesized into speech.
    - `speaker_wav` (str): The path or name of the speaker's audio file to be used for voice cloning.
    - `language` (str): The language code for the synthesis (e.g., 'en' for English, 'es' for Spanish).

    ### Returns:
    - A `FileResponse` containing the generated speech as a u-law audio file.
    """
    try:
        # Generar el archivo de salida WAV usando TTS
        output_file_path = XTTS.process_tts_to_file(
            text=request.text,
            speaker_name_or_path=request.speaker_wav,
            language=request.language.lower(),
            file_name_or_path=f'{str(uuid4())}.wav'
        )
        
        # Convertir el archivo WAV a u-law con tasa de 8kHz
        # Cargar el archivo WAV generado
        audio = AudioSegment.from_wav(output_file_path)
        
        # Cambiar la tasa de muestreo a 8kHz y convertir a u-law
        ulaw_file_path = f"{str(uuid4())}.ulaw"
        audio.set_frame_rate(8000).set_channels(1).set_sample_width(2).export(
            ulaw_file_path, format="wav", codec="pcm_mulaw"
        )
        
        # Si no se usa caché, se elimina el archivo después de usarlo
        if not XTTS.enable_cache_results:
            background_tasks.add_task(os.unlink, output_file_path)
            background_tasks.add_task(os.unlink, ulaw_file_path)

        # Retornar el archivo en formato u-law
        return FileResponse(
            path=ulaw_file_path,
            media_type='audio/basic',  # audio/basic para u-law según RFC 2046
            filename="output.ulaw"
        )

    except Exception as e:
        logger.error(f"Error processing TTS to u-law: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tts_ulaw64")
async def tts_to_audio_ulaw(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Convert text into speech, encode the resulting audio in u-law format with an 8 kHz sample rate, 
    and return the audio as a base64-encoded string with .ulaw extension.

    ### Parameters:
    - `request` (SynthesisRequest): Contains the following fields:
        - `text` (str): The text to be synthesized into speech.
        - `speaker_wav` (str): The path or name of the speaker's audio file to be used for voice cloning.
        - `language` (str): The language code for the synthesis (e.g., 'en' for English, 'es' for Spanish).

    ### Returns:
    - `JSONResponse`: A JSON object containing:
        - `audio_base64` (str): The base64-encoded u-law audio data.
        - `file_extension` (str): The file extension for the audio file, set to "ulaw".
        - `mime_type` (str): The MIME type for the u-law audio, set to "audio/basic".

    ### Example:
    - Request:
    ```json
    {
        "text": "Hello, world!",
        "speaker_wav": "speaker1.wav",
        "language": "en"
    }
    ```

    - Response:
    ```json
    {
        "audio_base64": "UExhdGVkIGJhc2U2NCBlbmNvZGVkIGRhdGEuLi4=",
        "file_extension": "ulaw",
        "mime_type": "audio/basic"
    }
    ```

    ### Error Handling:
    - If there is an error during the synthesis process or audio conversion, the function will return a `500 Internal Server Error` with a detailed message.

    ### Notes:
    - The function first generates a WAV file using the TTS model, then converts it into u-law format with an 8 kHz sample rate.
    - The resulting audio is encoded into base64 for transmission over HTTP, with `.ulaw` as the file extension and `audio/basic` as the MIME type.
    """
    try:
        # Generar el archivo de salida WAV usando TTS
        output_file_path = XTTS.process_tts_to_file(
            text=request.text,
            speaker_name_or_path=request.speaker_wav,
            language=request.language.lower(),
            file_name_or_path=f'{str(uuid4())}.wav'
        )
        
        # Convertir el archivo WAV a u-law
        # Cargar el archivo WAV generado
        audio = AudioSegment.from_wav(output_file_path)
        
        # Cambiar la tasa de muestreo a 8kHz y convertir a u-law
        ulaw_io = io.BytesIO()  # Crear un buffer en memoria para exportar el audio
        audio.set_frame_rate(8000).set_channels(1).set_sample_width(1).export(
            ulaw_io, format="ulaw"
        )
        
        # Obtener los datos binarios del archivo u-law desde el buffer en memoria
        ulaw_data = ulaw_io.getvalue()
        
        # Codificar los datos a base64 para la respuesta
        ulaw_base64 = base64.b64encode(ulaw_data).decode('utf-8')

        # Retornar los datos en formato base64 con extensión y formato .ulaw
        return JSONResponse(content={
            "audio_base64": ulaw_base64,
            "file_extension": "ulaw",
            "mime_type": "audio/basic"  # MIME type para archivos u-law
        })

    except Exception as e:
        logger.error(f"Error processing TTS to u-law: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002)
    except Exception as e:
        logger.exception("Error al iniciar la aplicación: %s", e)
        raise
