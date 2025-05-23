#before running , put the following in the terminal:
#source whisper_env/bin/activate    

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import signal
import pandas as pd
from datetime import datetime
import glob

import sys

# Set the microphone device index here
MICROPHONE_DEVICE_INDEX = 0  # 0 = MacBook Air-Mikrofon

# Configuration
MAX_RECORDINGS = 50  # Maximum number of recordings before stopping
CLEANUP_OLD_FILES = True  # Whether to delete old audio files
AUDIO_FOLDER = "backup/audio_files"

# Print available audio devices
print("Available audio devices:")
print(sd.query_devices())

def cleanup_old_files(folder=AUDIO_FOLDER, keep_last=5):
    """Clean up old audio files, keeping only the most recent ones."""
    if not os.path.exists(folder):
        return
    
    # Get all wav files
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    # Sort by modification time (newest first)
    wav_files.sort(key=os.path.getmtime, reverse=True)
    
    # Delete old files
    for old_file in wav_files[keep_last:]:
        try:
            os.remove(old_file)
            print(f"Cleaned up old file: {old_file}")
        except Exception as e:
            print(f"Error cleaning up {old_file}: {e}")

def signal_handler(sig, frame):
    print("\nStopping the recording process...")
    sys.exit(0)

def record_audio(duration=10, sample_rate=16000):
    """Record audio for a specified duration."""
    print(f"\nRecording for {duration} seconds...")
    print("Get ready to speak...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("Please speak into your microphone...")
    
    # Initialize recording array
    recording = np.zeros((int(duration * sample_rate), 1), dtype='float32')
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=MICROPHONE_DEVICE_INDEX) as stream:
        for i in range(0, int(duration * sample_rate), sample_rate):
            chunk, _ = stream.read(sample_rate)
            recording[i:i+sample_rate] = chunk
            
            # Calculate and print audio level
            level = np.abs(chunk).mean()
            print(f"Audio level: {level:.4f}", end='\r')
            
    print("\nRecording finished!")
    
    # Check if we got any audio
    avg_level = np.abs(recording).mean()
    if avg_level < 0.01:
        print("\n⚠️  WARNING: Very low audio levels detected!")
        print("Please try:")
        print("1. Speaking louder")
        print("2. Moving closer to the microphone")
        print("3. Checking your system's microphone settings")
        print(f"Current audio level: {avg_level:.4f}")
    
    return recording

def save_audio(recording, sample_rate=16000, folder=AUDIO_FOLDER):
    """Save the recording to a WAV file in the specified folder with a timestamped filename."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"recording_{timestamp}.wav")
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
    model = whisper.load_model("tiny")
    
    # Print selected audio device
    device_info = sd.query_devices(MICROPHONE_DEVICE_INDEX)
    print(f"\nUsing audio device: {device_info['name']}")
    print(f"Input channels: {device_info['max_input_channels']}")
    
    # DataFrame to store transcriptions
    df = pd.DataFrame(columns=["audio_file", "transcription"])
    
    try:
        for i in range(MAX_RECORDINGS):
            print(f"\n{'='*50}")
            print(f"Recording {i+1} of {MAX_RECORDINGS}")
            print(f"{'='*50}")
            
            try:
                # Record audio
                recording = record_audio(duration=10)  # Record for 10 seconds
                
                # Save the recording
                audio_file = save_audio(recording)
                print(f"Saved audio to: {audio_file}")
                
                # Transcribe the recording
                print("\nTranscribing...")
                transcription = transcribe_audio(model, audio_file)
                
                # Print the transcription
                print("\nTranscription:", transcription if transcription.strip() else "[No speech detected]")
                
                # Store in DataFrame
                df = pd.concat([df, pd.DataFrame({"audio_file": [audio_file], "transcription": [transcription]})], ignore_index=True)
                print(f"Added transcription to DataFrame. Current size: {len(df)}")
                
                # Clean up old files if enabled
                if CLEANUP_OLD_FILES:
                    cleanup_old_files(keep_last=MAX_RECORDINGS)
                
                # Wait a bit before next recording
                if i < MAX_RECORDINGS - 1:  # Don't wait after the last recording
                    print("\nWaiting 2 seconds before next recording...")
                    time.sleep(2)
                
            except Exception as e:
                print(f"\nError during recording {i+1}: {str(e)}")
                print("Continuing with next recording...")
                continue
            
    except KeyboardInterrupt:
        print("\nStopping the recording process...")
    finally:
        # Save DataFrame to CSV
        if not df.empty:
            csv_path = os.path.join(AUDIO_FOLDER, "transcriptions.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nTranscriptions saved to {csv_path}")
            print(f"Total recordings processed: {len(df)}")
        else:
            print("\nNo recordings were processed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
    