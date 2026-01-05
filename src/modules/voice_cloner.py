import torch
import torchaudio
import os
from chatterbox.tts import ChatterboxTTS

# Monkey-patch torchaudio if needed for Windows/ffmpeg interaction
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["ffmpeg", "soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

class VoiceCloner:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VoiceCloner initialized on {self.device} (models will load on-demand)")
        
        # LAZY LOADING: Don't load models until actually needed
        # This saves 3-4GB of RAM and prevents Windows paging file errors
        self.model = None  # Chatterbox (for English) - loads on first use
        self.coqui_model = None  # Coqui XTTS (for other languages) - loads on first use

    def generate_speech(self, text, reference_audio_path, language="en", output_path="output.wav", emotion="default"):
        """
        Generates speech using Chatterbox (En) or Coqui XTTS (Ur/Hi/etc).
        
        Args:
            text (str): Text to synthesize.
            reference_audio_path (str): Path to the reference audio file for cloning.
            language (str): Language code. 'en' uses Chatterbox, others use Coqui.
            output_path (str): Path to save the generated audio.
            emotion (str): Emotion tag (ignored by Coqui/Chatterbox, used for logging).
        """
        print(f"Generating speech [{language}] for: '{text[:20]}...'")
        
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        # English -> Chatterbox (with lazy loading and fallback)
        if language == 'en':
            # Lazy load Chatterbox on first English request
            if self.model is None:
                try:
                    print(f"Loading Chatterbox TTS model on {self.device}...")
                    self.model = ChatterboxTTS.from_pretrained(device=self.device)
                    print("✅ Chatterbox loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load Chatterbox: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    
                    # Check for specific memory error
                    if "paging file" in str(e).lower() or "os error 1455" in str(e).lower():
                        print("\n⚠️  MEMORY ERROR DETECTED:")
                        print("   Your Windows virtual memory (paging file) is too small.")
                        print("   Falling back to Coqui XTTS for English (uses less RAM).\n")
                    
                    # Mark as failed so we don't retry
                    self.model = "FAILED"
            
            # Try using Chatterbox if loaded
            if self.model and self.model != "FAILED":
                try:
                    wav = self.model.generate(text, audio_prompt_path=reference_audio_path)
                    torchaudio.save(output_path, wav, self.model.sr)
                    return output_path
                except Exception as e:
                    print(f"Chatterbox generation failed: {e}")
                    print("Falling back to Coqui XTTS for this segment...")
            
            # Fallback: Use Coqui for English if Chatterbox failed
            if self.model == "FAILED" or self.model is None:
                print("Using Coqui XTTS for English as fallback...")
                language = 'en'  # Coqui supports English too
                # Continue to Coqui section below

        # Other Languages (or English fallback) -> Coqui TTS (XTTS v2)
        # This handles all non-English languages OR English if Chatterbox failed
        try:
            if self.coqui_model is None:
                print(f"Loading Coqui XTTS v2 for language '{language}'...")
                from TTS.api import TTS
                # Load XTTS v2 (auto-downloads if needed)
                self.coqui_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                print("✅ Coqui XTTS loaded successfully")
            
            # XTTS Generation
            self.coqui_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio_path,
                language=language,
                split_sentences=True
            )
            return output_path
            
        except ImportError:
            print("[ERROR] 'TTS' library not installed. Please run: pip install TTS")
            raise
        except Exception as e:
            print(f"Coqui XTTS generation failed: {e}")
            raise

if __name__ == "__main__":
    pass
