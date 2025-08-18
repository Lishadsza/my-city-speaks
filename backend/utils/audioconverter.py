import subprocess
import os

def convert_to_wav(input_path):
    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e)
        return None
