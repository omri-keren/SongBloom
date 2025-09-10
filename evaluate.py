import os
import glob
import csv
import numpy as np
from frechet_audio_distance import FrechetAudioDistance

# Paths
GENERATED_DIR = "./wav_output"              # Your generated audio
REFERENCE_DIR = "./maestro_wav_16k"     # Cleaned & resampled MAESTRO clips
REFERENCE_STATS_NPZ = "maestro_stats.npz"
CSV_OUTPUT = "fad_results.csv"

# Collect generated .wav files
# generated_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.wav")))

# if not generated_files:
#     raise RuntimeError(f"No generated .wav files found in {GENERATED_DIR}")

# Initialize FAD
fad = FrechetAudioDistance(
    model_name="vggish",
    use_pca=False,
    verbose=True,
)

# # If .npz does not exist, create it from MAESTRO reference files
# if not os.path.exists(REFERENCE_STATS_NPZ):
#     print("📦 Computing reference stats from MAESTRO .wav files...")
#     reference_files = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.wav")))
#     if not reference_files:
#         raise RuntimeError(f"No reference .wav files found in {REFERENCE_DIR}")
#     embeddings = fad.get_embeddings(reference_files)
#     mu, sigma = fad.calculate_embd_statistics(embeddings)
#     np.savez(REFERENCE_STATS_NPZ, mu=mu, sigma=sigma)
#     print(f"✅ Saved reference stats to: {REFERENCE_STATS_NPZ}")
# else:
#     print(f"📁 Using existing stats: {REFERENCE_STATS_NPZ}")

# Compute FAD score
gen_path = '/home/cs/omriker/SongBloom/wav_output_2'
ref_dir = '/home/cs/omriker/SongBloom/wav_output'
score = fad.score(gen_path,ref_dir)

# Save result
with open(CSV_OUTPUT, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["reference_stats", "generated_dir", "fad_score"])
    writer.writerow([REFERENCE_STATS_NPZ, GENERATED_DIR, score])

print(f"✅ FAD score (vs MAESTRO): {score}")
print(f"💾 Results saved to: {CSV_OUTPUT}")
