import os
import subprocess
import sys
import gdown
from src.config import Config

class LipSyncer:
    def __init__(self):
        self.wav2lip_dir = os.path.join(Config.BASE_DIR, "src", "Wav2Lip")
        self.weights_dir = os.path.join(self.wav2lip_dir, "checkpoints")
        self.model_path = os.path.join(self.weights_dir, "wav2lip_gan.pth")
        
        self._setup_wav2lip()

    def _setup_wav2lip(self):
        """
        Clones Wav2Lip repo and downloads weights if missing.
        """
        # 1. Clone Repo
        if not os.path.exists(os.path.join(self.wav2lip_dir, "inference.py")):
            print("Cloning Wav2Lip repository...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", self.wav2lip_dir],
                    check=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to clone Wav2Lip: {e}")
                print("Please install git or manually download Wav2Lip to src/Wav2Lip")
                raise

        # 2. Download Weights
        os.makedirs(self.weights_dir, exist_ok=True)
        if not os.path.exists(self.model_path):
            print(f"Downloading Wav2Lip GAN Model to {self.model_path}...")
            # Using a HuggingFace mirror for easier direct download than GDrive
            url = "https://huggingface.co/ln312/wav2lip_gan/resolve/main/wav2lip_gan.pth"
            try:
                # We use gdown or standard request. Since gdown is installed:
                output = self.model_path
                subprocess.run(
                    ["gdown", url, "-O", output],
                    check=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to download model: {e}")
                # Fallback to standard request if gdown fails on HF url
                import requests
                response = requests.get(url, stream=True)
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

    def sync_lips(self, video_path, audio_path, output_path):
        """
        Runs Wav2Lip inference.
        """
        print(f"Starting Lip Sync: {video_path} + {audio_path}...")
        
        inference_script = os.path.join(self.wav2lip_dir, "inference.py")
        
        # Wav2Lip Argument Construction
        cmd = [
            sys.executable, inference_script,
            "--checkpoint_path", self.model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "1", # 1 = 720p/1080p usually, higher numbers downscale more
            "--nosmooth" # Often cleaner results for dubbing
        ]
        
        try:
            # We must run this inside the Wav2Lip directory context usually? 
            # Or just pass absolute paths. Wav2Lip imports 'models', so PYTHONPATH needs to include it.
            
            env = os.environ.copy()
            # Add Wav2Lip dir to PYTHONPATH so it can find its submodules
            env["PYTHONPATH"] = self.wav2lip_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            subprocess.run(cmd, env=env, check=True)
            
            if not os.path.exists(output_path):
                raise FileNotFoundError("Wav2Lip finished but output file is missing.")
                
            print(f"Lip Sync Complete: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"Wav2Lip Inference Failed: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] Lip Sync Error: {e}")
            raise

if __name__ == "__main__":
    pass
