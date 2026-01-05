import argparse
import os
from src.orchestrator import Orchestrator
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="AI Video Dubbing Orchestrator")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--lang", default=None, help="Target language code (e.g., es, fr, de, it)")
    parser.add_argument("--tone", default=None, help="Tone preference (optional)")
    parser.add_argument("--service", default=None, help="Translation service: 'openrouter', 'mistral', 'google'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return

    # Interactive Prompt for Language if not explicitly provided (or if using default blindly)
    # To detect if user provided it, we can default to None in argparse
    # Priority: CLI Arg > Config > Interactive Prompt
    target_lang = args.lang
    
    if target_lang is None:
        if Config.TARGET_LANGUAGE:
            target_lang = Config.TARGET_LANGUAGE
            print(f"Using default target language from config: '{target_lang}'")
        else:
            target_lang = input("Enter target language code (e.g., 'en' ,'es', 'fr', 'de', 'it', 'hi'): ").strip()
            
    if not target_lang:
        print("No language selected. Defaulting to 'es' (Spanish).")
        target_lang = 'es'

    orchestrator = Orchestrator()
    
    try:
        final_path = orchestrator.run_pipeline(
            video_path=args.video_path,
            target_language=target_lang,
            tone_preference=args.tone,
            translation_service=args.service
        )
        print(f"\nSUCCESS! Dubbed video saved to:\n{final_path}")
    except Exception as e:
        print(f"\nFATAL ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
