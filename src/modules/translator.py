import os
import json
import time
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
from deep_translator import GoogleTranslator
from src.config import Config

class Translator:
    def __init__(self, target_language="en", service_override=None):
        self.target_languge = target_language
        
        # Determine service: explicit override > config > default
        self.service = service_override if service_override else Config.TRANSLATION_SERVICE
        self.client = None
        self.model = None

        print(f"Initializing Translator Service: {self.service}")

        # Initialize fallback translator (always available)
        self.fallback_translator = GoogleTranslator(source='auto', target=self.target_languge)

        if self.service == "google":
            print("Selected Google Translate (Non-LLM).")
            return

        # Setup LLM Clients
        if self.service == "mistral":
            self.api_key = Config.MISTRAL_API_KEY
            self.model = Config.MISTRAL_MODEL
            base_url = "https://api.mistral.ai/v1"
            
            if not self.api_key:
                print("[WARNING] MISTRAL_API_KEY not found. Switching to Google Fallback.")
            else:
                self.client = OpenAI(base_url=base_url, api_key=self.api_key)

        elif self.service == "openrouter":
            self.api_key = Config.OPENROUTER_API_KEY
            self.model = Config.OPENROUTER_MODEL
            base_url = "https://openrouter.ai/api/v1"
            
            if not self.api_key:
                print("[WARNING] OPENROUTER_API_KEY not found. Switching to Google Fallback.")
            else:
                self.client = OpenAI(base_url=base_url, api_key=self.api_key)
        
        else:
             print(f"[WARNING] Unknown service '{self.service}'. Defaulting to Google Fallback.")

    def _use_fallback_translation(self, segments):
        print(f"⚠️ Switching to Google Translate fallback for {len(segments)} segments...")
        translated_segments = []
        for seg in segments:
            new_seg = seg.copy()
            try:
                # deep_translator is blocking, but fast enough for fallback
                translated = self.fallback_translator.translate(seg["text"])
                new_seg["text_translated"] = translated
            except Exception as e:
                print(f"[ERROR] Google Translate failed for segment: {e}")
                new_seg["text_translated"] = seg["text"] # Ultimate fallback
            translated_segments.append(new_seg)
        return translated_segments

    def translate_segments(self, segments):
        """
        Translates a list of segments using an LLM for context awareness.
        Falls back to Google Translate on error.
        """
        if not self.client:
            print("[INFO] No LLM API Key. Using Google Translate directly.")
            return self._use_fallback_translation(segments)

        print(f"Translating {len(segments)} segments to {self.target_languge} via LLM ({self.model})...")
        
        # Prepare the payload
        # Minimal payload to save tokens/complexity
        transcript_text = "\n".join([f"{i}: [Duration: {seg.get('duration', 0):.2f}s] {seg['text']}" for i, seg in enumerate(segments)])
        
        system_prompt = (
            f"You are a professional dubbing translator. Translate the following transcript lines to {self.target_languge}. "
            "CRITICAL: The translation must be concise to match the original speaking duration. "
            "If the target language naturally takes longer, shorten the phrasing or omit filler words while keeping the core meaning. "
            "The input format is 'Line ID: [Duration: Xs] Text'. "
            "Return ONLY a JSON array of strings, where each string is the translated text for the corresponding line number. "
            "Example: [\"Hola\", \"Mundo\"]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript_text}
                ],
                # response_format={"type": "json_object"} 
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown blocks
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            translated_list = json.loads(content)
            
            if len(translated_list) != len(segments):
                print(f"[WARNING] Mismatch in translation count ({len(translated_list)} vs {len(segments)}). Fallback mapping.")
            
            translated_segments = []
            for i, seg in enumerate(segments):
                new_seg = seg.copy()
                if i < len(translated_list):
                    new_seg["text_translated"] = translated_list[i]
                else:
                    new_seg["text_translated"] = seg["text"] # Fallback
                translated_segments.append(new_seg)

            return translated_segments

        except (RateLimitError, AuthenticationError, APIConnectionError) as e:
            print(f"\n[LLM ERROR] {type(e).__name__}: {e}")
            print("  -> Initiating Fallback to Google Translate...\n")
            return self._use_fallback_translation(segments)

        except Exception as e:
            print(f"[ERROR] LLM Translation failed: {e}")
            print("  -> Fallback: Using Google Translate.")
            return self._use_fallback_translation(segments)

if __name__ == "__main__":
    # Test
    # tl = Translator(target_language="es")
    # print(tl.translate_segment("Hello, this is a test of the translation system."))
    pass
