import os
import subprocess
import shutil
import glob
from pydub import AudioSegment
from src.config import Config

class AudioSeparator:
    def __init__(self, output_dir="temp"):
        self.output_dir = output_dir
        self.separation_out_dir = os.path.join(self.output_dir, "separated")
        os.makedirs(self.separation_out_dir, exist_ok=True)

    def separate(self, audio_path, bgm_output_path=None):
        """
        Separates audio into vocals and background music (drums + bass + other).
        Returns a tuple: (vocals_path, bgm_path)
        """
        print(f"Separating audio: {audio_path}...")
        
        # We use demucs via subprocess to avoid complex dependency management within python if possible,
        # but importing it is also fine. CLI is often more robust for simple usage.
        # Command: demucs --two-stems=vocals -n htdemucs -o <output_dir> <input_file>
        # --two-stems=vocals will produce 'vocals.wav' and 'no_vocals.wav' (which is the BGM)
        
        # Note: 'htdemucs' is the default and fast model.
        
        try:
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-n", Config.DEMUCS_MODEL,
                "-o", self.separation_out_dir,
                audio_path
            ]
            
            # Run Demucs
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Demucs output structure: <output_dir>/<model_name>/<filename_no_ext>/...
            filename_no_ext = os.path.basename(audio_path).split('.')[0]
            # Demucs output structure: <output_dir>/<model_name>/<filename_no_ext>/...
            filename_no_ext = os.path.basename(audio_path).split('.')[0]
            model_name = Config.DEMUCS_MODEL
            
            # Demucs might replace spaces with underscores, or keep them. We check both.
            result_dir = os.path.join(self.separation_out_dir, model_name, filename_no_ext)
            if not os.path.exists(result_dir):
                # Try sanitized version
                sanitized_name = filename_no_ext.replace(" ", "_")
                result_dir_sanitized = os.path.join(self.separation_out_dir, model_name, sanitized_name)
                if os.path.exists(result_dir_sanitized):
                    result_dir = result_dir_sanitized
                    print(f"  [DEBUG] Found result dir with sanitized name: {result_dir}")

            vocals_path = os.path.join(result_dir, "vocals.wav")
            generated_bgm_path = os.path.join(result_dir, "no_vocals.wav")
            
            print(f"  [DEBUG] Looking for BGM at: {generated_bgm_path}")
            
            final_bgm_path = generated_bgm_path
            
            # If a specific BGM output path is requested (e.g. into the 'bgm' folder)
            if bgm_output_path:
                print(f"Saving BGM to library: {bgm_output_path}")
                if os.path.exists(generated_bgm_path):
                    shutil.copy2(generated_bgm_path, bgm_output_path)
                    final_bgm_path = bgm_output_path
                else:
                    print(f"  [ERROR] Source BGM file not found: {generated_bgm_path}")
            
            if not os.path.exists(vocals_path) or not os.path.exists(generated_bgm_path):
                # Try finding ANY wav file in result_dir just in case naming changed
                found_files = os.listdir(result_dir) if os.path.exists(result_dir) else []
                print(f"  [DEBUG] Files in result dir: {found_files}")
                raise FileNotFoundError(f"Demucs failed to produce expected output files in {result_dir}")
                
            print(f"Separation complete.")
            print(f"  Vocals: {vocals_path}")
            print(f"  BGM: {final_bgm_path}")
            
            return vocals_path, final_bgm_path

        except subprocess.CalledProcessError as e:
            print(f"Demucs Error: {e.stderr.decode() if e.stderr else str(e)}")
            # Fallback: return original as vocals, None as BGM (or silence)
            print("Fallback: Using original audio as vocals, no BGM extraction.")
            return audio_path, None
        except Exception as e:
            print(f"Separation Exception: {e}")
            return audio_path, None

if __name__ == "__main__":
    # Test stub
    pass
