import imageio_ffmpeg
import subprocess
from pathlib import Path

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg at: {ffmpeg_exe}")

data_dir = Path("baby-data")

for label in ["low", "medium", "high"]:
    input_file = data_dir / f"{label}.mp4"
    output_file = data_dir / f"{label}.wav"
    
    result = subprocess.run([
        ffmpeg_exe, "-i", str(input_file),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_file), "-y"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {label}.mp4 → {label}.wav")
    else:
        print(f"✗ {label} failed: {result.stderr[-200:]}")