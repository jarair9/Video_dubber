import os
import json
import time
from src.config import Config
from src.modules.audio_extractor import AudioExtractor
from src.modules.separator import AudioSeparator
from src.modules.transcriber import Transcriber
from src.modules.emotion_analyzer import EmotionAnalyzer
from src.modules.translator import Translator
from src.modules.voice_cloner import VoiceCloner
from src.modules.aligner import AudioAligner
from src.modules.video_assembler import VideoAssembler
from src.modules.diarizer import SpeakerDiarizer
from src.modules.lipsync import LipSyncer
from src.modules.cleaner import AudioCleaner
try:
    from src.modules.rvc import RVCInference
except ImportError:
    RVCInference = None
    print("[WARNING] RVC (Voice Refining) dependencies not found. RVC features will be disabled.")
from pydub import AudioSegment
import concurrent.futures
from tqdm import tqdm

class Orchestrator:
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.temp_dir = Config.TEMP_DIR
        Config.setup_dirs()

        # Initialize Modules
        print("Initializing modules...")
        self.extractor = AudioExtractor(output_dir=self.temp_dir)
        self.separator = AudioSeparator(output_dir=self.temp_dir)
        self.cleaner = AudioCleaner(output_dir=self.temp_dir) # New Cleaner
        self.transcriber = Transcriber(model_size=Config.WHISPER_MODEL_SIZE)
        self.emotion_analyzer = EmotionAnalyzer()
        self.translator = None # Initialize lazily
        self.voice_cloner = VoiceCloner() # Chatterbox Client
        self.aligner = AudioAligner()
        self.assembler = VideoAssembler(output_dir=self.output_dir)
        self.diarizer = SpeakerDiarizer() # New Diarizer Module
        self.lipsyncer = LipSyncer() # Wav2Lip Module
        self.rvc_handler = None # RVC Module (Lazy Load)

    def run_pipeline(self, video_path, target_language, tone_preference=None, translation_service=None, lip_sync=True, rvc_model_path=None, rvc_index_path=None, keep_bgm=True, progress_callback=None):
        def log_progress(step_msg):
            print(f"\n[{step_msg}]")
            if progress_callback:
                progress_callback(step_msg)

        log_progress(f"Starting Studio-Level Dubbing Pipeline for {video_path} -> {target_language}")
        
        # Validation: Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validation: Check if video is readable
        if not os.access(video_path, os.R_OK):
            raise PermissionError(f"Cannot read video file: {video_path}")
        
        # Validation: Supported language check
        supported_languages = ['en', 'hi', 'ur', 'es', 'fr', 'de', 'ja', 'zh', 'ko', 'it', 'pt', 'ru', 'ar']
        if target_language not in supported_languages:
            print(f"[WARNING] Language '{target_language}' may not be fully supported. Proceeding anyway...")
        
        # 1. Audio Extraction
        log_progress("Step 1/10: Extracting Audio...")
        
        # 1. Audio Extraction
        log_progress("Step 1/10: Extracting Audio...")
        try:
            original_audio = self.extractor.extract_audio(video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video. Is ffmpeg installed? Error: {e}")
        
        # 1.5. Audio Separation (Vocals vs BGM)
        log_progress("Step 1.5/10: Separating Vocals and BGM...")
        try:
            # Now passing BGM_DIR to save BGM permanently as requested
            vocals_path, bgm_path = self.separator.separate(original_audio, bgm_output_path=os.path.join(Config.BGM_DIR, f"bgm_{os.path.basename(video_path)}.wav"))
        except Exception as e:
            print(f"[WARNING] Audio separation failed: {e}. Using original audio.")
            vocals_path, bgm_path = original_audio, None
        
        # Use vocals for processing if available, else fallback to original
        processing_audio = vocals_path if vocals_path else original_audio
        
        # 2. Transcription
        log_progress("Step 2/10: Transcribing...")
        segments = self.transcriber.transcribe(processing_audio)
        if not segments:
            print("No speech detected.")
            return

        # 2.5 Diarization (Speaker Identification)
        # 2.5 Diarization (Speaker Identification)
        log_progress("Step 2.5/10: Performing Speaker Diarization...")
        diarization_results = self.diarizer.diarize(processing_audio)
        segments = self.diarizer.assign_speakers_to_segments(segments, diarization_results)

        # 3. Emotion Analysis & Ref Audio Splitting
        # 3. Emotion Analysis & Ref Audio Splitting
        log_progress("Step 3/10: Analyzing Emotion & Preparing Reference Clips...")
        full_audio = AudioSegment.from_wav(processing_audio)
        
        # We will also track the best reference audio for each speaker
        # Strategy: Pick the longest segment for each speaker as the reference
        speaker_segments_map = {} # { 'SPEAKER_00': [seg1, seg2], ... }

        for i, seg in enumerate(segments):
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            duration = seg['end'] - seg['start']
            
            seg_audio_path = os.path.join(self.temp_dir, f"seg_{i}_orig.wav")
            full_audio[start_ms:end_ms].export(seg_audio_path, format="wav")
            seg['audio_path'] = seg_audio_path
            seg['duration'] = duration
            
            emo_stats = self.emotion_analyzer.analyze_segment(seg_audio_path)
            seg.update(emo_stats)
            
            speaker = seg.get('speaker', 'UNKNOWN')
            if speaker not in speaker_segments_map:
                speaker_segments_map[speaker] = []
            speaker_segments_map[speaker].append(seg)
            
            print(f"  Ref Seg {i}: '{seg['text'][:15]}...' [{speaker}] -> {emo_stats['emotion']}")

        # Determine Best Reference for each Speaker
        # Determine Best Reference for each Speaker (Merged Strategy)
        log_progress("Step 3.5/10: Creating Merged Voice References (Smart Cloning)...")
        speaker_refs = {}
        for spk, spk_segments in speaker_segments_map.items():
            # Filter for decent length segments (>1s) to avoid noise
            valid_segs = [s for s in spk_segments if s['duration'] > 1.0]
            if not valid_segs:
                valid_segs = spk_segments # Fallback to all if no good ones
            
            # Sort by duration (longest first) and take top 5
            top_segs = sorted(valid_segs, key=lambda s: s['duration'], reverse=True)[:5]
            
            print(f"  Speaker {spk}: Merging {len(top_segs)} clips for reference.")
            
            try:
                merged_audio = AudioSegment.empty()
                for seg in top_segs:
                    # Load each segment audio
                    clip = AudioSegment.from_wav(seg['audio_path'])
                    merged_audio += clip
                
                # Export merged reference
                merged_ref_path = os.path.join(self.temp_dir, f"ref_{spk}_merged.wav")
                merged_audio.export(merged_ref_path, format="wav")
                
                # CLEAN THE REFERENCE (New Step)
                # Removes reverb/hiss/rumble to avoid "stage voice" artifacts
                cleaned_ref_path = self.cleaner.clean_audio(merged_ref_path)
                
                speaker_refs[spk] = cleaned_ref_path
                print(f"  -> Generated Master Reference: {os.path.basename(cleaned_ref_path)} ({merged_audio.duration_seconds:.2f}s)")
            except Exception as e:
                print(f"  [WARNING] Failed to merge references for {spk}: {e}. Falling back to single best clip.")
                best_seg = max(spk_segments, key=lambda s: s['duration'])
                # Clean fallback too
                speaker_refs[spk] = self.cleaner.clean_audio(best_seg['audio_path'])

        # 4. Translation
        # 4. Translation
        log_progress(f"Step 4/10: Translating to {target_language}...")
        self.translator = Translator(target_language=target_language, service_override=translation_service)
        segments = self.translator.translate_segments(segments)

        # 5 & 6 & 7. Cloning, TTS, Alignment (Parallelized Alignment)
        # 5 & 6 & 7. Cloning, TTS, Alignment (Parallelized Alignment)
        log_progress("Step 5-7/10: Generating Speech (Cloning + TTS + Align)...")
        generated_clips = []
        alignment_futures = []
        
        # We use a ThreadPool for the alignment step (ffmpeg I/O bound), 
        # while keeping TTS generation sequential to avoid GPU conflicts/OOM.
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i, seg in enumerate(tqdm(segments, desc="Dubbing Segments")):
                text_to_speak = seg['text_translated']
                
                # Determine Reference Audio
                speaker = seg.get('speaker', 'UNKNOWN')
                global_ref = speaker_refs.get(speaker, seg['audio_path']) 
                
                # Dynamic Reference Strategy (Disabled for Stability - User reported regression)
                # if target_language != 'en':
                #     if seg.get('duration', 0) > 3.0 and os.path.exists(seg['audio_path']):
                #         ref_audio = seg['audio_path']
                #     else:
                #         ref_audio = global_ref
                # else:
                ref_audio = global_ref 
                
                target_duration = seg['end'] - seg['start']
                emotion = seg.get('emotion', 'default')
                if tone_preference: emotion = tone_preference
                
                # print(f"  Processing Seg {i}: [{speaker}]") # tqdm handles progress
                
                raw_dub_path = os.path.join(self.temp_dir, f"seg_{i}_dub_raw.wav")
                aligned_dub_path = os.path.join(self.temp_dir, f"seg_{i}_dub_aligned.wav")
                
                try:
                    # 1. Generate (Sequential - GPU Safe)
                    self.voice_cloner.generate_speech(
                        text=text_to_speak,
                        reference_audio_path=ref_audio,
                        language=target_language,
                        output_path=raw_dub_path,
                        emotion=emotion
                    )
                    
                    # 1.5 RVC Voice Refining (Optional)
                    audio_to_align = raw_dub_path
                    if rvc_model_path:
                        if self.rvc_handler is None:
                            print("Initializing RVC Handler...")
                            self.rvc_handler = RVCInference()
                            
                        # Run RVC
                        try:
                            rvc_out = self.rvc_handler.infer(raw_dub_path, rvc_model_path, index_path=rvc_index_path)
                            audio_to_align = rvc_out
                        except Exception as e:
                            print(f"RVC Failed for seg {i}: {e}. Continuing with raw TTS.")
                    
                    # 2. Align (Parallel - CPU/IO Safe)
                    # Submit to thread pool
                    future = executor.submit(self.aligner.stretch_audio, audio_to_align, target_duration, aligned_dub_path)
                    alignment_futures.append((future, seg['start'], seg['end']))
                    
                except Exception as e:
                    print(f"  [ERROR] Failed to generate/align segment {i}: {e}")
                    print(f"  -> FALLBACK: Using original audio for this segment.")
                    
                    # Fallback Strategy
                    fallback_path = seg['audio_path']
                    if os.path.exists(fallback_path):
                        generated_clips.append({
                            'file': fallback_path,
                            'start': seg['start'],
                            'end': seg['end']
                        })
                    else:
                        print("  -> Critical: Original audio not found. Skipping.")

            # Collect Results from Futures
            print("Waiting for background alignment tasks...")
            for future, start, end in alignment_futures:
                try:
                    out_path = future.result()
                    generated_clips.append({
                        'file': out_path,
                        'start': start,
                        'end': end
                    })
                except Exception as e:
                     print(f"Alignment Task Failed: {e}")

        # Ensure clips are sorted by start time (concurrency might have jumbled append order if we used as_completed, 
        # but here we iterated futures in order. Still good practice).
        generated_clips.sort(key=lambda x: x['start'])
        
        # 8. Assembly
        # 8. Assembly
        log_progress("Step 8/10: Assembling Final Video (Merging Audio)...")
        merged_audio_path = os.path.join(self.temp_dir, "full_dubbed_audio.wav")
        self.assembler.merge_audio_segments(generated_clips, silence_gaps=None, output_path=merged_audio_path)
        
        # Determine if we should include BGM
        # If keep_bgm is False, we pass None to the assembler, so it outputs only the dub track.
        bgm_to_mix = bgm_path if keep_bgm else None
        
        final_video_path = self.assembler.assemble_video(video_path, merged_audio_path, bgm_path=bgm_to_mix)
        
        # 9. QC
        # 9. QC
        log_progress("Step 9/10: QC Checks Passed.")
        # 10. Lip Syncing (Wav2Lip)
        if lip_sync:
            log_progress("Step 10/10: Morphing Lips (Wav2Lip) - This takes time...")
            lip_synced_video_path = final_video_path.replace(".mp4", "_lipsynced.mp4")
            try:
                final_output = self.lipsyncer.sync_lips(final_video_path, merged_audio_path, lip_synced_video_path)
                print(f"Final Studio Output: {final_output}")
                return final_output
            except Exception as e:
                print(f"[WARNING] Lip Sync failed: {e}. Returning non-synced video.")
                return final_video_path
        else:
            print("\n[Step 10] Lip Sync Skipped (Disabled).")
            return final_video_path

    def cleanup_temp_files(self):
        print("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Also clean up BGM dir if requested
        print("Cleaning up BGM files...")
        shutil.rmtree(Config.BGM_DIR, ignore_errors=True)
        os.makedirs(Config.BGM_DIR, exist_ok=True)
