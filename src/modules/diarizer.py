import os
import torch
from pyannote.audio import Pipeline
from src.config import Config

class SpeakerDiarizer:
    def __init__(self, auth_token=None):
        self.auth_token = auth_token or os.environ.get("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
        if Config.USE_DIARIZATION:
            print(f"Initializing Speaker Diarizer on {self.device}...")
            
            if not self.auth_token:
                print("\n[WARNING] HF_TOKEN is missing! Speaker Diarization will be SKIPPED.")
                print("  -> To enable: Set HF_TOKEN in .env for pyannote.audio")
                print("  -> Continuing in Mono-Speaker Mode (all segments = SPEAKER_00)\n")
                self.pipeline = None
            else:
                try:
                    # Note: 'pyannote/speaker-diarization-3.1' requires acceptance of user conditions on HF
                    # and a valid token. 
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.auth_token
                    )
                    if self.pipeline:
                        self.pipeline.to(self.device)
                    else:
                        print("Warning: Could not load Diarization pipeline. Check HF_TOKEN and model access.")
                except Exception as e:
                    print(f"\n[ERROR] Failed to initialize Diarization pipeline: {e}")
                    print("  -> Check if you accepted the model license on Hugging Face: https://huggingface.co/pyannote/speaker-diarization-3.1")
                    print("  -> Continuing without diarization (fallback to mono-speaker).\n")
                    self.pipeline = None

    def diarize(self, audio_path):
        """
        Returns a list of segments with speaker labels.
        Format: [{'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'}, ...]
        """
        if not self.pipeline:
            print("Diarizer not loaded. Skipping.")
            return []

        print(f"Diarizing {audio_path}...")
        try:
            diarization = self.pipeline(audio_path)
            
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Brief check
            unique_speakers = set(s['speaker'] for s in results)
            print(f"  Found {len(unique_speakers)} speakers: {unique_speakers}")
            
            return results
        except Exception as e:
            print(f"Diarization error: {e}")
            return []

    def assign_speakers_to_segments(self, transcription_segments, diarization_results):
        """
        Matches transcription segments (which have text) to the most likely speaker 
        from the diarization results based on time overlap.
        """
        if not diarization_results:
            # Fallback: Assign everyone to 'SPEAKER_00' if no diarization
            for seg in transcription_segments:
                seg['speaker'] = 'SPEAKER_00'
            return transcription_segments

        print("Assigning speakers to transcription segments...")
        
        for seg in transcription_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            seg_dur = seg_end - seg_start
            
            # Find all diarization turns that overlap with this segment
            overlaps = []
            for d_seg in diarization_results:
                # Calculate overlap duration
                overlap_start = max(seg_start, d_seg['start'])
                overlap_end = min(seg_end, d_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    overlaps.append((d_seg['speaker'], overlap))
            
            if not overlaps:
                seg['speaker'] = "UNKNOWN"
            else:
                # Pick the speaker with the most overlap duration
                # Sum overlap per speaker (in case of fragmented diarization)
                speaker_totals = {}
                for spk, duration in overlaps:
                    speaker_totals[spk] = speaker_totals.get(spk, 0) + duration
                
                # Best speaker
                best_speaker = max(speaker_totals, key=speaker_totals.get)
                seg['speaker'] = best_speaker
                
            # print(f"  Segment ({seg_start:.1f}-{seg_end:.1f}) -> {seg['speaker']}")
            
        return transcription_segments
