import os
import subprocess
import sys
import requests
from src.config import Config

class RVCInference:
    def __init__(self):
        self.rvc_dir = os.path.join(Config.BASE_DIR, "src", "RVC")
        self.assets_dir = os.path.join(self.rvc_dir, "assets")
        self.hubert_path = os.path.join(self.assets_dir, "hubert", "hubert_base.pt")
        self.rmvpe_path = os.path.join(self.assets_dir, "rmvpe", "rmvpe.pt")
        
        self._setup_rvc()

    def _setup_rvc(self):
        """
        Clones RVC repo and downloads base models (Hubert, RMVPE).
        """
        # 1. Clone Repo
        if not os.path.exists(os.path.join(self.rvc_dir, "infer_cli.py")):
            print("Cloning RVC (Retrieval-based Voice Conversion) repository...")
            try:
                # Cloning a lightweight fork or the main one. 
                # The main one is huge. Let's try to find a CLI-friendly version or use the main one but shallow.
                # Use RVC-Project main for stability.
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", self.rvc_dir],
                    check=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to clone RVC: {e}")
                raise

        # 2. Download Base Models (Hubert & RMVPE)
        # Check if assets exist
        os.makedirs(os.path.dirname(self.hubert_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.rmvpe_path), exist_ok=True)

        if not os.path.exists(self.hubert_path):
            print("Downloading Hubert Base Model (Required for RVC)...")
            self._download_file("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt", self.hubert_path)

        if not os.path.exists(self.rmvpe_path):
            print("Downloading RMVPE Model (Required for Pitch)...")
            self._download_file("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt", self.rmvpe_path)

    def _download_file(self, url, output_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def infer(self, input_audio, model_path, index_path=None, f0_up_key=0, method="rmvpe"):
        """
        Runs RVC Inference on the input audio.
        """
        if not model_path or not os.path.exists(model_path):
            print(f"[WARNING] RVC Model path invalid: {model_path}. Skipping RVC.")
            return input_audio

        print(f"Running RVC Inference on {os.path.basename(input_audio)}...")
        output_path = input_audio.replace(".wav", "_rvc.wav")
        
        # We need to call the RVC inference script. 
        # The main repo usually has 'tools/infer_cli.py' or we can invoke 'infer_batch_rvc' via python.
        # Since the interface changes, we might need to rely on a specific script in the repo.
        # Let's assume we can construct a python script on the fly to import RVC and run it, 
        # OR run a CLI command if available. 
        
        # Strategy: Run a custom python snippet that sets up the path and calls the RVC core.
        # This is safer than relying on 'infer_cli.py' which might not handle arguments cleanly.
        
        rvc_script = f"""
import sys
import os
sys.path.append(r"{self.rvc_dir}")
from vc_infer_pipeline import VC
from config import Config
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid
import torch

# ... (This requires complex imports from the RVC repo)
# Simplified: use tools/infer_cli.py if it exists, or similar
"""
        # For robustness, let's look for known CLI hooks in the repo.
        # RVC-Project usually has 'tools/infer_cli.py'.
        
        script_path = os.path.join(self.rvc_dir, "tools", "infer_cli.py")
        if not os.path.exists(script_path):
            # Fallback for newer versions where file structure changed
            script_path = os.path.join(self.rvc_dir, "infer_cli.py")
            
        cmd = [
            sys.executable, script_path,
            "--input_path", input_audio,
            "--output_path", output_path,
            "--model_path", model_path,
            "--f0_up_key", str(f0_up_key),
            "--f0_method", method,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--is_half", "True" if torch.cuda.is_available() else "False"
        ]
        
        if index_path and os.path.exists(index_path):
            cmd.extend(["--index_path", index_path])
            
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.rvc_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            subprocess.run(cmd, env=env, check=True)
            
            if os.path.exists(output_path):
                print(f"RVC Success: {output_path}")
                return output_path
            else:
                print("RVC finished but output missing.")
                return input_audio
                
        except Exception as e:
            print(f"RVC Inference Failed: {e}")
            return input_audio

if __name__ == "__main__":
    pass
