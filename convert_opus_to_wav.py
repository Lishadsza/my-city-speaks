import os
import subprocess

# Exact full path to ffmpeg.exe
ffmpeg_path = r"C:\ffmpeg-7.1.1-essentials_build (1)\ffmpeg\bin\ffmpeg.exe"

# Input/output folders
input_folder = "recordings"
output_folder = "converted_wav"
os.makedirs(output_folder, exist_ok=True)

# Convert .opus to .wav
for filename in os.listdir(input_folder):
    if filename.endswith(".opus"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace(".opus", ".wav")
        output_path = os.path.join(output_folder, output_filename)

        command = [
            ffmpeg_path,
            "-y",              # Overwrite existing files
            "-i", input_path,  # Input file
            output_path        # Output file
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted: {filename} → {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {filename}: {e}")
