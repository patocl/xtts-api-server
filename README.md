
# XTTS-API Server

A simple FastAPI server to host XTTSv2, built with support for Text-to-Speech (TTS) functionalities, including real-time audio streaming, support for multiple speakers, and model switching.

This project is a **modified version of [xtts-api-server](https://github.com/daswer123/xtts-api-server)** and utilizes [XTTSv2](https://github.com/coqui-ai/TTS). It was created to support SillyTavern, but can be used for other purposes as well.

Feel free to contribute or modify the code for your own needs.

## Changelog

Track all changes on the [release page](https://github.com/AdrianXira/xtts-api-server/releases).

## Installation

### Clone the repository

To install this project, you must first clone the repository from GitHub:

```bash
git clone https://github.com/AdrianXira/xtts-api-server.git
cd xtts-api-server
```

### Set up the environment

1. **Create a Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux
   venv\Scripts\activate  # Windows
   ```

2. **Install the dependencies**:
   You can install all the necessary dependencies from the `pyproject.toml` and `requirements.txt` files provided.
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional dependencies**:
   If you are using a GPU, you may want to install PyTorch with CUDA support to improve performance.
   ```bash
   pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

### Manual Installation

Alternatively, you can install the project manually by following these steps:

```bash
# Clone the repository
git clone https://github.com/AdrianXira/xtts-api-server.git
cd xtts-api-server
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate in Windows
# Install dependencies
pip install -r requirements.txt
# Install PyTorch for GPU support
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# Install the project
pip install .
# Launch the server
python -m xtts_api_server
```

## Using Docker

A `Dockerfile` is provided for containerized deployment. Additionally, a `docker-compose.yml` is available for easier management.

### Build and Run with Docker Compose

1. **Build the Docker image**:
   ```bash
   docker compose build
   ```

2. **Run the container**:
   ```bash
   docker compose up -d  # Runs the server in the background
   ```

#### Changes

##### .env File

Added .env file to specify ports for Docker Compose and dockerfile dinamically.

.env
   ```bash
   # Define the internal port for the service
   PORT=8020

   # Define the external port for the service (host)
   HOST_PORT=8080
   ```

##### Added build.ps1

File for Windows users to build and run the Docker image.

2. **Run and deploy the container**:

If the docker is running it will be stopped and removed if the build is successful. If not it will be built and run.

   ```bash
   .\build.ps1    # Runs the server in the background
   ```
##### dockerfile improved

Removed extra layers and added a .dockerignore file to exclude unnecessary files from the build context.

##### docker-compose.yml improved

Export volumens for folders model, output and speakers (previosly named examples), to facilitate the management of new speakers.


## Running the Server

Once installed, you can start the server with the following command:

```bash
python -m xtts_api_server
```

By default, the server will run on `localhost:8020`.

If you want to allow external access, use the `--listen` option:

```bash
python -m xtts_api_server --listen
```

### Example of Command-line Usage

You can pass various flags to customize the behavior of the server:

```
usage: xtts_api_server [-h] [-hs HOST] [-p PORT] [-sf SPEAKER_FOLDER] [-o OUTPUT] [-t TUNNEL_URL] [-ms MODEL_SOURCE] [--listen] [--use-cache] [--lowvram] [--deepspeed] [--streaming-mode] [--stream-play-sync]
```

### Key Options:
- `--deepspeed`: Speeds up processing significantly (2-3x).
- `--listen`: Allows external access to the server.
- `--streaming-mode`: Enables streaming mode for real-time playback.

For a full list of options, run `python -m xtts_api_server -h`.


### Endpoints

- **`/tts_stream`**: Streams audio from provided text in real-time.
- **`/tts_to_audio`**: Generates and returns a `.wav` audio file from text.
- **`/tts_ulaw`**: Generates and returns audio in u-law format (8 kHz).
- **`/tts_ulaw64`**: Returns the u-law audio in Base64 encoding.
- **`/get_speakers`**: Lists available speaker files for voice cloning.

## Adding Speakers

You can add speaker `.wav` files to the `speakers/` folder. This allows you to clone voices from audio files. Simply place the `.wav` file in the folder, and the API will use it for synthesis.

### Example Folder Structure:

```bash
xtts-server/
    ├── speakers/
    │   ├── speaker1.wav
    │   ├── speaker2.wav
    └── models/
        └── tts-model.tar.gz
```

## Using Your Own Model

You can load custom models by placing them in the `models/` folder. Ensure that the folder contains the required files:

- `config.json`
- `vocab.json`
- `model.pth`

## Starting Server

`python -m xtts_api_server` will run on default ip and port (localhost:8020)

Use the `--deepspeed` flag to process the result fast ( 2-3x acceleration )

```
usage: xtts_api_server [-h] [-hs HOST] [-p PORT] [-sf SPEAKER_FOLDER] [-o OUTPUT] [-t TUNNEL_URL] [-ms MODEL_SOURCE] [--listen] [--use-cache] [--lowvram] [--deepspeed] [--streaming-mode] [--stream-play-sync]

Run XTTSv2 within a FastAPI application

options:
  -h, --help show this help message and exit
  -hs HOST, --host HOST
  -p PORT, --port PORT
  -d DEVICE, --device DEVICE `cpu` or `cuda`, you can specify which video card to use, for example, `cuda:0`
  -sf SPEAKER_FOLDER, --speaker-folder The folder where you get the samples for tts
  -o OUTPUT, --output Output folder
  -mf MODELS_FOLDERS, --model-folder Folder where models for XTTS will be stored, finetuned models should be stored in this folder
  -t TUNNEL_URL, --tunnel URL of tunnel used (e.g: ngrok, localtunnel)
  -ms MODEL_SOURCE, --model-source ["api","apiManual","local"]
  -v MODEL_VERSION, --version You can download the official model or your own model, official version you can find [here](https://huggingface.co/coqui/XTTS-v2/tree/main)  the model version name is the same as the branch name [v2.0.2,v2.0.3, main] etc. Or you can load your model, just put model in models folder
  --listen Allows the server to be used outside the local computer, similar to -hs 0.0.0.0
  --use-cache Enables caching of results, your results will be saved and if there will be a repeated request, you will get a file instead of generation
  --lowvram The mode in which the model will be stored in RAM and when the processing will move to VRAM, the difference in speed is small
  --deepspeed allows you to speed up processing by several times, automatically downloads the necessary libraries
  --streaming-mode Enables streaming mode, currently has certain limitations, as described below.
  --streaming-mode-improve Enables streaming mode, includes an improved streaming mode that consumes 2gb more VRAM and uses a better tokenizer and more context.
  --stream-play-sync Additional flag for streaming mod that allows you to play all audio one at a time without interruption
```

You can specify the path to the file as text, then the path counts and the file will be voiced

You can load your own model, for this you need to create a folder in models and load the model with configs, note in the folder should be 3 files `config.json` `vocab.json` `model.pth`

If you want your host to listen, use -hs 0.0.0.0 or use --listen

The -t or --tunnel flag is needed so that when you get speakers via get you get the correct link to hear the preview. More info [here](https://imgur.com/a/MvpFT59)

Model-source defines in which format you want to use xtts:

1. `local` - loads version 2.0.2 by default, but you can specify the version via the -v flag, model saves into the models folder and uses `XttsConfig` and `inference`.
2. `apiManual` - loads version 2.0.2 by default, but you can specify the version via the -v flag, model saves into the models folder and uses the `tts_to_file` function from the TTS api
3. `api` - will load the latest version of the model. The -v flag won't work.

All versions of the XTTSv2 model can be found [here](https://huggingface.co/coqui/XTTS-v2/tree/main)  the model version name is the same as the branch name [v2.0.2,v2.0.3, main] etc.

The first time you run or generate, you may need to confirm that you agree to use XTTS.

# About Streaming mode

Streaming mode allows you to get audio and play it back almost immediately. However, it has a number of limitations.

You can see how this mode works [here](https://www.youtube.com/watch?v=jHylNGQDDA0) and [here](https://www.youtube.com/watch?v=6vhrxuWcV3U)

Now, about the limitations

1. Can only be used on a local computer
2. Playing audio from the your pc
3. Does not work endpoint `tts_to_file` only `tts_to_audio` and it returns 1 second of silence.

You can specify the version of the XTTS model by using the `-v` flag.

Improved streaming mode is suitable for complex languages such as Chinese, Japanese, Hindi or if you want the language engine to take more information into account when processing speech.

`--stream-play-sync` flag - Allows you to play all messages in queue order, useful if you use group chats. In SillyTavern you need to turn off streaming to work correctly

# API Docs

API Docs can be accessed from [http://localhost:8020/docs](http://localhost:8020/docs)

# How to add speaker

By default the `speakers` folder should appear in the folder, you need to put there the wav file with the voice sample, you can also create a folder and put there several voice samples, this will give more accurate results

# Note on creating samples for quality voice cloning

The following post is a quote by user [Material1276 from reddit](https://www.reddit.com/r/Oobabooga/comments/1807tsl/comment/ka5l8w9/?share_id=_5hh4KJTXrEOSP0hR0hCK&utm_content=2&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1)

> Some suggestions on making good samples
>
> Keep them about 7-9 seconds long. Longer isn't necessarily better.
>
> Make sure the audio is down sampled to a Mono, 22050Hz 16 Bit wav file. You will slow down processing by a large % and it seems cause poor quality results otherwise (based on a few tests). 24000Hz is the quality it outputs at anyway!
>
> Using the latest version of Audacity, select your clip and Tracks > Resample to 22050Hz, then Tracks > Mix > Stereo to Mono. and then File > Export Audio, saving it as a WAV of 22050Hz
>
> If you need to do any audio cleaning, do it before you compress it down to the above settings (Mono, 22050Hz, 16 Bit).
>
> Ensure the clip you use doesn't have background noises or music on e.g. lots of movies have quiet music when many of the actors are talking. Bad quality audio will have hiss that needs clearing up. The AI will pick this up, even if we don't, and to some degree, use it in the simulated voice to some extent, so clean audio is key!
>
> Try make your clip one of nice flowing speech, like the included example files. No big pauses, gaps or other sounds. Preferably one that the person you are trying to copy will show a little vocal range. Example files are in [here](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/coqui_tts/voices)
>
> Make sure the clip doesn't start or end with breathy sounds (breathing in/out etc).
>
> Using AI generated audio clips may introduce unwanted sounds as its already a copy/simulation of a voice, though, this would need testing.

# Credit

1. Thanks to the author **Kolja Beigel** for the repository [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) , I took some of its code for my project.
2. Thanks **[erew123](https://github.com/oobabooga/text-generation-webui/issues/4712#issuecomment-1825593734)** for the note about creating samples and the code to download the models
3. Thanks **lendot** for helping to fix the multiprocessing bug and adding code to use multiple samples for speakers
