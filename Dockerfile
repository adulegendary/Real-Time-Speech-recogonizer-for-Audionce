FROM python:3.12-slim

WORKDIR /app

# System deps: gcc for building pyaudio/webrtcvad, PortAudio for audio, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    alsa-utils \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so resemblyzer doesn't pull in CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/google_credentials

EXPOSE 5000

CMD ["python", "src/main.py"]
