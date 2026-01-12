import torch
import soundfile as sf
from transformers import AutoProcessor, VitsModel

MODEL_NAME = "facebook/mms-tts-eng"

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME)

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input text
text = "Hello. This text to speech example now works correctly."

# Tokenize text
inputs = processor(text=text, return_tensors="pt").to(device)

# Generate speech
with torch.no_grad():
    outputs = model(**inputs)

# Extract waveform
audio = outputs.waveform.cpu().numpy().squeeze()

print(audio.min(), audio.max())


# Save audio
sf.write("output.wav", audio, samplerate=16000)

print("Audio saved as output.wav")
