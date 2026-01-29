import subprocess
import winsound
from pathlib import Path


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "piperVoices"

MODEL = MODEL_DIR / "de_DE-thorsten-medium.onnx"
CONFIG = MODEL_DIR / "de_DE-thorsten-medium.onnx.json"
OUTPUT_WAV = BASE_DIR / "output.wav"

def text_to_speech(text: str):
    if not text.strip():
        raise ValueError("Text ist leer")

    if not MODEL.exists():
        raise FileNotFoundError("ONNX-Modell fehlt")
    if not CONFIG.exists():
        raise FileNotFoundError("JSON-Konfig fehlt")

    cmd = [
        "piper",
        "--model", str(MODEL),
        "--config", str(CONFIG),
        "--output_file", str(OUTPUT_WAV)
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True
    )

    process.communicate(text)

    if not OUTPUT_WAV.exists():
        raise RuntimeError("Audio wurde nicht erzeugt")

    winsound.PlaySound(str(OUTPUT_WAV), winsound.SND_FILENAME)