import os
import subprocess
import ffmpeg
import json
from pydub import AudioSegment, silence

class AudioExtractor:
    def __init__(self, output_dir="temp"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_audio(self, video_path):
        """
        Extracts audio from video file and saves it as a WAV file.
        Returns the path to the extracted audio file.
        """
        filename = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(self.output_dir, f"{filename}.wav")
        
        print(f"Extracting audio from {video_path} to {output_path}...")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, ac=1, ar=16000) # Mono, 16kHz for ML models
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode() if e.stderr else str(e))
            raise

    def detect_silence(self, audio_path, min_silence_len=500, silence_thresh=-40):
        """
        Detects silent chunks in the audio.
        Returns a list of [start, end] timestamps in milliseconds.
        """
        audio = AudioSegment.from_wav(audio_path)
        silence_ranges = silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        return silence_ranges

    def get_audio_info(self, audio_path):
        """Returns duration and other metadata."""
        probe = ffmpeg.probe(audio_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        return audio_stream

if __name__ == "__main__":
    # Test stub
    extractor = AudioExtractor()
    # print(extractor.extract_audio("test_video.mp4"))
