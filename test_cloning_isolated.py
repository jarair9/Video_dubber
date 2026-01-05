import os
from src.modules.voice_cloner import VoiceCloner
import ffmpeg

def test_cloning_isolated():
    video_path = "video.mp4"
    ref_audio_path = "temp/test_ref_clip.wav"
    output_path = "outputs/test_cloned_voice.wav"
    
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"--- Isolated Voice Cloning Test ---")
    
    # 1. content extraction (Need a short reference clip)
    if os.path.exists(video_path):
        print(f"Extracting 5s reference audio from {video_path}...")
        try:
            (
                ffmpeg
                .input(video_path, ss=0, t=5) # Take first 5 seconds
                .output(ref_audio_path, ac=1, ar=16000)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg extraction failed: {e.stderr.decode()}")
            return
    else:
        print(f"Video file {video_path} not found!")
        # Create dummy if needed, but per user request we expect video.mp4
        return

    # 2. Initialize Cloner
    try:
        vc = VoiceCloner() # Uses .env token
    except Exception as e:
        print(f"Failed to initialize VoiceCloner: {e}")
        return

    # 3. Generate
    text = "Hello! This is a test of the local voice cloning system."
    print(f"Generating: '{text}'")
    
    try:
        vc.generate_speech(
            text=text,
            reference_audio_path=ref_audio_path,
            output_path=output_path
            # language/emotion are ignored by SpeechT5 but kept for interface compatibility
        )
        print(f"\nSUCCESS! Cloned audio saved to: {output_path}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_cloning_isolated()
