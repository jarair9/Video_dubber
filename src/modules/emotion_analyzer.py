import torch
import librosa
import numpy as np
import os
from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        print(f"Loading Emotion model '{model_name}'...")
        # Using the pipeline for audio classification
        self.classifier = pipeline("audio-classification", model=model_name)

    def analyze_emotion(self, audio_path):
        """
        Detects primary emotion from audio segment.
        Returns: {emotion: str, confidence: float}
        """
        try:
            # The pipeline handles loading audio, but we might need to handle short segments
            # or sampling rate. The mode usually expects 16kHz.
            predictions = self.classifier(audio_path)
            # predictions is a list of dicts: [{'score': 0.1, 'label': 'happy'}, ...]
            primary = max(predictions, key=lambda x: x['score'])
            return primary['label'], primary['score']
        except Exception as e:
            print(f"Emotion analysis failed for {audio_path}: {e}")
            return "neutral", 0.0

    def analyze_prosody(self, audio_path):
        """
        Extracts pitch, energy, speaking rate.
        Note: Speaking rate needs text length, so this returns raw audio features.
        """
        y, sr = librosa.load(audio_path, sr=None)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)
        energy_level = "medium"
        if avg_energy < 0.01: energy_level = "low"
        elif avg_energy > 0.05: energy_level = "high"
        
        # Pitch (F0)
        # Using pyin for robustness on speech
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_f0 = f0[~np.isnan(f0)]
        avg_pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 0
        
        pitch_tendency = "mid" 
        # Simple heuristic thresholds (can be improved with gender detection)
        if avg_pitch < 150: pitch_tendency = "low"
        elif avg_pitch > 250: pitch_tendency = "high"

        return {
            "energy": energy_level,
            "pitch": pitch_tendency,
            "avg_pitch_hz": float(avg_pitch),
            "avg_energy_val": float(avg_energy)
        }

    def analyze_segment(self, audio_path):
        emotion, conf = self.analyze_emotion(audio_path)
        prosody = self.analyze_prosody(audio_path)
        
        return {
            "emotion": emotion,
            "confidence": conf,
            **prosody
        }

if __name__ == "__main__":
    # Test
    # ea = EmotionAnalyzer()
    # print(ea.analyze_segment("temp/test_segment.wav"))
    pass
