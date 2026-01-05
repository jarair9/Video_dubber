import ffmpeg
import os
import math

class AudioAligner:
    def __init__(self):
        pass

    def stretch_audio(self, audio_path, target_duration, output_path):
        """
        Time-stretches the audio to match the target duration.
        Uses ffmpeg 'atempo' filter.
        """
        # Get current duration
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
        
        if duration == 0:
            return audio_path # Safety check
            
        rate = duration / target_duration
        
        # FFmpeg atempo filter is limited to 0.5 to 2.0. 
        # If rate is outside we need to chain them, but for dubbing, 
        # drastic changes > 2.0 are bad anyway.
        
        # Clamp rate for safety/quality
        if rate < 0.5: rate = 0.5
        if rate > 2.0: rate = 2.0
        
        print(f"Time-stretching {audio_path}: {duration:.2f}s -> {target_duration:.2f}s (Rate: {rate:.2f}x)")
        
        try:
            (
                ffmpeg
                .input(audio_path)
                .filter('atempo', rate)
                .output(output_path)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            print(f"Alignment Error: {e.stderr.decode()}")
            return audio_path # Fallback to original

    def align_segments(self, audio_segments_map):
        """
        Iterates through generated assignments and aligns them.
        audio_segments_map: list of dicts { 'file': path, 'target_duration': float }
        """
        aligned_files = []
        for item in audio_segments_map:
            out_name = item['file'].replace(".wav", "_aligned.wav")
            self.stretch_audio(item['file'], item['target_duration'], out_name)
            aligned_files.append(out_name)
        return aligned_files

if __name__ == "__main__":
    pass
