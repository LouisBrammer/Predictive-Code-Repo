#before running , put the following in the terminal:
#source whisper_env/bin/activate    

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import signal

import sys

# Print available audio devices
print("Available audio devices:")
print(sd.query_devices())

def signal_handler(sig, frame):
    print("\nStopping the recording process...")
    sys.exit(0)

def record_audio(duration=10, sample_rate=16000):
    """Record audio for a specified duration."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=0)
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    return recording

def save_audio(recording, sample_rate=16000, filename="temp_recording.wav"):
    """Save the recording to a WAV file."""
    # Convert float32 to int16
    recording = (recording * 32767).astype(np.int16)
    wav.write(filename, sample_rate, recording)
    return filename

def transcribe_audio(model, audio_file):
    """Transcribe the audio file using Whisper."""
    print("Transcribing...")
    result = model.transcribe(audio_file)
    return result["text"]

def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load the Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    while True:
        try:
            # Record audio
            recording = record_audio(duration=10)  # Record for 10 seconds
            
            # Save the recording
            audio_file = save_audio(recording)
            
            # Transcribe the recording
            transcription = transcribe_audio(model, audio_file)
            
            # Print the transcription
            print("\nTranscription:", transcription)
            print("\nPress Ctrl+C to stop recording")
            
            # Clean up the temporary file
            os.remove(audio_file)
            
            # Wait a bit before next recording
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nStopping the recording process...")
            sys.exit(0)

if __name__ == "__main__":
    main()
    