import ffmpeg
import os

class VideoAssembler:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def assemble_video(self, original_video_path, dubbed_audio_path, bgm_path=None, output_filename="dubbed_video.mp4"):
        """
        Replaces audio in the video with the new dubbed track.
        If bgm_path is provided, mixes it with the dubbed audio.
        """
        output_path = os.path.join(self.output_dir, output_filename)
        print(f"Assembling video: {output_path}...")
        
        video_input = ffmpeg.input(original_video_path)
        video = video_input.video
        dub_audio = ffmpeg.input(dubbed_audio_path)
        
        try:
            if bgm_path and os.path.exists(bgm_path):
                print(f"Mixing BGM from {bgm_path} with Smart Ducking...")
                bgm_audio = ffmpeg.input(bgm_path)
                
                # IMPLEMENTING SIDECHAIN DUCKING
                # We want BGM to lower volume when Dub starts.
                # [bgm][dub] sidechaincompress [bgm_ducked]
                # Then [dub][bgm_ducked] amix [out]
                
                # Parameters:
                # threshold: trigger level (e.g. 0.015)
                # ratio: compression ratio (e.g. 2)
                # attack: 20ms
                # release: 1000ms (1s fade back in)
                
                ducked_bgm = ffmpeg.filter(
                    [bgm_audio, dub_audio],
                    'sidechaincompress',
                    threshold=0.05, 
                    ratio=8, 
                    attack=50, 
                    release=800
                )
                
                # Mix the ducked BGM with the Dub
                # Using duration='first' (Video length usually matches Dub track length logic roughly, 
                # or we rely on video input to cut the stream). 
                # Ideally, we want the shortest of video vs audio?
                # Usually we want to preserve the full dubbed audio length if it matches video.
                
                mixed_audio = ffmpeg.filter([dub_audio, ducked_bgm], 'amix', inputs=2, duration='first')
                
                (
                    ffmpeg
                    .output(video, mixed_audio, output_path, vcodec='copy', acodec='aac')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                # No BGM, just replace audio
                (
                    ffmpeg
                    .output(video, dub_audio, output_path, vcodec='copy', acodec='aac')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
            print("Video assembly complete.")
            return output_path
        except ffmpeg.Error as e:
            print(f"Assembly Error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def merge_audio_segments(self, segment_files, silence_gaps, output_path="full_dub.wav"):
        """
        Concatenates audio segments with silence in between to create the full track.
        This is a simplified approach; accurate placement requires a timeline filter.
        """
        # A more robust way is to create a complex filter in ffmpeg placing inputs at timestamps.
        # But constructing a long complex filter string can be error-prone.
        
        # Pydub is better for this specific task of timeline construction
        from pydub import AudioSegment
        
        full_track = AudioSegment.silent(duration=0)
        current_time = 0
        
        # This assumes segment_files is a list of tuples/dicts: (file_path, start_ms, end_ms)
        # Sort by start time just in case
        segment_files.sort(key=lambda x: x['start'])
        
        canvas = AudioSegment.silent(duration=int(segment_files[-1]['end'] * 1000) + 1000) # Canvas size (ms)
        
        for seg in segment_files:
            path = seg['file']
            start_ms = seg['start'] * 1000 # Convert to ms
            
            # Load segment
            audio_chunk = AudioSegment.from_wav(path)
            
            # Overlay on canvas
            canvas = canvas.overlay(audio_chunk, position=start_ms)
            
        canvas.export(output_path, format="wav")
        return output_path

if __name__ == "__main__":
    pass
