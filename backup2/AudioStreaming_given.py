import whisper
import numpy as np
import queue
import threading
import sounddevice as sd
import time

# Load Whisper model
model = whisper.load_model("tiny")

# Setup audio parameters
CHUNK = 1024
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3  # Process in 3-second chunks

# Initialize audio queue
audio_queue = queue.Queue()

def record_audio():
    recording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    return recording

def process_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            # Process with Whisper
            result = model.transcribe(audio_data)
            print(f"Transcription: {result['text']}")

# Start processing thread
threading.Thread(target=process_audio, daemon=True).start()

# Main recording loop
try:
    print("* Recording started - speak into the microphone")
    while True:
        # Record audio for RECORD_SECONDS
        audio_data = record_audio()
        # Put audio chunk in queue for processing
        audio_queue.put(audio_data)
        
except KeyboardInterrupt:
    print("* Recording stopped")

# Clean up nn
sd.stop()
sd.terminate()   