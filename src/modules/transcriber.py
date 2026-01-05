import whisper
import torch
import json
import os

class Transcriber:
    def __init__(self, model_size="base", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path):
        """
        Transcribes the audio file.
        Returns a list of segments with start, end, text, and basic speaker placeholder.
        """
        print(f"Transcribing {audio_path}...")
        result = self.model.transcribe(audio_path)
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "speaker": "Speaker 0", # Whisper standard doesn't do diarization without extra tools
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
            
        return segments

    def save_transcription(self, segments, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2)
        print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    # Test stub
    # tx = Transcriber()
    # print(tx.transcribe("temp/test.wav"))
    pass
