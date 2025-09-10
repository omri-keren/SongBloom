import os, glob, pathlib, sys
import soundfile as sf
import numpy as np
from tqdm import tqdm

try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except ImportError:
    HAVE_TORCHAUDIO = False
    from scipy.signal import resample_poly

# ------- paths -------
MAESTRO_ROOT = pathlib.Path("/home/cs/omriker/SongBloom/maestro-v3.0.0")
OUTPUT_DIR   = pathlib.Path("/home/cs/omriker/SongBloom/maestro_wav_16k")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SR = 16000
MAX_FILES = 50                     # change or set None for all

# ------- collect wavs -------
wav_paths = sorted(MAESTRO_ROOT.glob("20*/**/*.wav"))   # all years, recursive
if not wav_paths:
    sys.exit("❌  No WAV files found; check MAESTRO_ROOT path")

if MAX_FILES:
    wav_paths = wav_paths[:MAX_FILES]

print(f"▶ Found {len(wav_paths)} WAVs, converting to {TARGET_SR} Hz mono…")

written = 0
for src_path in tqdm(wav_paths, unit="file"):
    audio, sr = sf.read(src_path, always_2d=False)

    # to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # resample if needed
    if sr != TARGET_SR:
        if HAVE_TORCHAUDIO:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, TARGET_SR
            ).numpy()
        else:                       # scipy fallback
            gcd = np.gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd)

    # scale float32 → int16 for smaller files
    audio = np.clip(audio, -1.0, 1.0)
    int_audio = (audio * 32767).astype(np.int16)

    dst_path = OUTPUT_DIR / src_path.name
    sf.write(dst_path, int_audio, TARGET_SR, subtype="PCM_16")
    written += 1

print(f"✅  Wrote {written} files to {OUTPUT_DIR.resolve()}")
