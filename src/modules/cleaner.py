import os
import noisereduce as nr
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import butter, lfilter
import librosa
import soundfile as sf
from src.config import Config

class AudioCleaner:
    def __init__(self, output_dir="temp"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_audio(self, audio_path, suffix="_clean"):
        """
        Applies cleaning pipeline:
        1. Spectral Gating (removes hiss/static)
        2. High-Pass Filter (removes rumble < 100Hz)
        """
        try:
            filename = os.path.basename(audio_path).split('.')[0]
            output_path = os.path.join(self.output_dir, f"{filename}{suffix}.wav")
            
            print(f"Cleaning audio: {filename}...")
            
            # Load audio using librosa (handles various formats better, resampling to 24k is fine for cloning)
            # But for noisereduce, maintaining sample rate is good.
            data, rate = librosa.load(audio_path, sr=None)
            
            # 1. Noise Reduction (Spectral Gating)
            # Assuming noise is stationary (like hiss), we can estimate it from the whole clip
            # prop_decrease=0.8 means remove 80% of noise (conservative to avoid artifacts)
            reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.75, stationary=True)
            
            # 2. High-Pass Filter (Remove Rumble)
            filtered_audio = self._highpass_filter(reduced_noise, cutoff=100, fs=rate)
            
            # Save
            sf.write(output_path, filtered_audio, rate)
            print(f"  -> Cleaned saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[WARNING] Audio cleaning failed: {e}. Returning original.")
            return audio_path

    def _highpass_filter(self, data, cutoff=100, fs=44100, order=5):
        try:
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            y = lfilter(b, a, data)
            return y
        except Exception as e:
            print(f"Filter error: {e}")
            return data

if __name__ == "__main__":
    # Test
    pass
