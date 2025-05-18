import whisper, sounddevice as sd, queue, threading

model = whisper.load_model("tiny")
q = queue.Queue()
def rec(): q.put(sd.rec(int(16000*3), samplerate=16000, channels=1, dtype='float32')); sd.wait()
def trans(): 
    while 1: 
        if not q.empty(): print(model.transcribe(q.get())['text'])
threading.Thread(target=trans, daemon=True).start()
print("Recording started - Ctrl+C to stop")
try:
    while 1: rec()
except KeyboardInterrupt: print("Stopped")