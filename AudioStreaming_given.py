import whisper
import numpy as np
import queue
import threading
import pyaudio
import time

# Load Whisper model
model = whisper.load_model("base")

# Setup audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
RECORD_SECONDS = 3  # Process in 3-second chunks

# Initialize PyAudio
p = pyaudio.PyAudio()
audio_queue = queue.Queue()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def process_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            # Convert audio to the format Whisper expects
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process with Whisper
            result = model.transcribe(audio_np)
            print(f"Transcription: {result['text']}")

# Start processing thread
threading.Thread(target=process_audio, daemon=True).start()

# Main recording loop
try:
    print("* Recording started - speak into the microphone")
    while True:
        # Collect audio for RECORD_SECONDS
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Put audio chunk in queue for processing
        audio_queue.put(b''.join(frames))
        
except KeyboardInterrupt:
    print("* Recording stopped")

# Clean up nn
stream.stop_stream()
stream.close()
p.terminate()   