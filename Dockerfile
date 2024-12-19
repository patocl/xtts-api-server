# Stage 1: Build Stage for dependencies
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS dependencies

# Install system dependencies
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

# Upgrade pip, install general Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel ninja virtualenv

# Install torch, torchaudio, and other dependencies
RUN python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install deepspeed

# Stage 2: Build Stage for application
FROM dependencies AS application

# Set workdir for application
WORKDIR /xtts-server

# Set ARG and ENV for port (application-specific)
ARG PORT=8020
ENV PORT=$PORT

# Expose the port
EXPOSE $PORT

# Create necessary directories (optional, these will be overwritten by volumes)
RUN mkdir -p /xtts-server/xtts-server/models /xtts-server/xtts-server/speakers /xtts-server/xtts-server/output

# Copy the requirements file and install Python dependencies (will use cache if not modified)
COPY requirements.txt pyproject.toml README.md /xtts-server/

RUN python3 -m pip install --use-deprecated=legacy-resolver -r /xtts-server/requirements.txt

# Copy application source code
COPY xtts_api_server /xtts-server/xtts_api_server/

# Install the application package
RUN pip install . \
    && python3 -m pip cache purge

# Command to run the application
CMD ["python3", "-m", "xtts_api_server", "--listen", "-p", "$PORT", "-t", "http://localhost:$PORT", "-sf", "xtts-server/speakers", "-o", "xtts-server/output", "-mf", "xtts-server/models", "--deepspeed"]
