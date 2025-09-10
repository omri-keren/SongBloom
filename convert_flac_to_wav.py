import os
import soundfile as sf
import scipy.io.wavfile as wavfile
import numpy as np
import librosa

INPUT_DIR = "./output"      # Directory with .flac files
OUTPUT_DIR = "./wav_output"     # Where to save .wav files
TARGET_SR = 16000               # Target sampling rate

def convert_file(input_path, output_path):
    data, sr = sf.read(input_path)
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Normalize if needed (safety)
    data = np.clip(data, -1.0, 1.0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write WAV using int16 format
    wavfile.write(output_path, sr, (data * 32767).astype(np.int16))
    print(f"✔ Converted: {input_path} → {output_path}")

def walk_and_convert(input_root, output_root):
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".flac"):
                rel_path = os.path.relpath(root, input_root)
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_root, rel_path, file.replace(".flac", ".wav"))
                convert_file(input_path, output_path)

if __name__ == "__main__":
    walk_and_convert(INPUT_DIR, OUTPUT_DIR)
    print("✅ All .flac files converted to 16kHz mono .wav.")

