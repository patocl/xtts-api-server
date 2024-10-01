# Use an official NVIDIA base image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set label for the docker image description
LABEL description="Docker image for xtts-api-server"

# Install required packages and clean up the apt cache to reduce image size
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-dev \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libportaudiocpp0 \
    git \
    python3 \
    python3-pip \
    make \
    g++ \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages in a single layer
RUN python3 -m pip install --upgrade pip setuptools wheel ninja virtualenv

# Create necessary directories
RUN mkdir -p /xtts-server/xtts-server/models /xtts-server/xtts-server/speakers /xtts-server/xtts-server/output

# Copy the male.wav file from the local example directory to the speakers directory in the container
COPY example/male.wav /xtts-server/xtts-server/speakers/male.wav
COPY example/Israel.wav /xtts-server/xtts-server/speakers/Israel.wav
COPY example/Jorge.wav /xtts-server/xtts-server/speakers/Jorge.wav
COPY example/Tinoco.wav /xtts-server/xtts-server/speakers/Tinoco.wav
COPY example/Elisa.wav /xtts-server/xtts-server/speakers/Elisa.wav
COPY example/LetIZZIa.wav /xtts-server/xtts-server/speakers/LetIZZIa.wav

# Copy the xtts_api_server directory to /xtts-server/xtts_api_server in the container
COPY xtts_api_server /xtts-server/xtts_api_server

# Set workdir for application
WORKDIR /xtts-server

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --use-deprecated=legacy-resolver -r requirements.txt

# Install torch, torchaudio, and other dependencies
RUN python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install deepspeed

# Install the xtts_api_server package using pip
COPY pyproject.toml /xtts-server/
COPY README.md /xtts-server/
RUN pip install .

# Clean up pip cache to reduce image size
RUN python3 -m pip cache purge

# Copy and extract the tar.gz model file into the models directory
COPY tts-model.tar.gz /xtts-server/xtts-server/models/tts-model.tar.gz
RUN tar -xzvf /xtts-server/xtts-server/models/tts-model.tar.gz -C /xtts-server/xtts-server/models/ 

# Expose the application port
EXPOSE 8020

# Command to run the xtts_api_server when the container starts
CMD ["python3", "-m", "xtts_api_server", "--listen", "-p", "8020", "-t", "http://localhost:8020", "-sf", "xtts-server/speakers", "-o", "xtts-server/output", "-mf", "xtts-server/models", "--deepspeed"]
