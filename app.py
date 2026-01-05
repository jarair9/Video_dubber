import gradio as gr
import os
import shutil
import traceback
from src.orchestrator import Orchestrator
from src.config import Config

# Ensure directories exist
Config.setup_dirs()

def run_dubbing(video_file, target_language, use_lipsync, include_bgm, tone_preference, rvc_model_file=None, progress=gr.Progress()):
    if not video_file:
        return None, "❌ Please upload a video file."
    
    video_path = video_file.name
    print(f"\n{'='*60}")
    print(f"Processing Video: {video_path}")
    print(f"Target Language: {target_language}")
    print(f"Lip Sync: {use_lipsync} | BGM: {include_bgm}")
    print(f"{'='*60}\n")
    
    # RVC Model handling
    rvc_path = rvc_model_file.name if rvc_model_file else None
    
    try:
        # Initialize Orchestrator
        print("Initializing orchestrator...")
        orchestrator = Orchestrator()
        
        # Progress Callback Wrapper with step tracking
        step_count = [0]  # Mutable counter
        total_steps = 10
        
        def update_progress(msg):
            step_count[0] += 1
            progress_pct = min(step_count[0] / total_steps, 0.95)  # Cap at 95% until complete
            progress(progress_pct, desc=f"[{step_count[0]}/{total_steps}] {msg}")
            print(f"\n>>> {msg}")
        
        print("\nStarting dubbing pipeline...")
        final_video = orchestrator.run_pipeline(
            video_path=video_path,
            target_language=target_language,
            tone_preference=tone_preference,
            lip_sync=use_lipsync,
            rvc_model_path=rvc_path,
            keep_bgm=include_bgm,
            progress_callback=update_progress
        )
        
        progress(1.0, desc="✅ Complete!")
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Final output: {final_video}")
        print(f"{'='*60}\n")
        
        return final_video, "✅ Dubbing Complete! Enjoy your studio-quality video."
        
    except Exception as e:
        # Log full error details to console
        print(f"\n{'='*60}")
        print(f"❌ ERROR OCCURRED:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        # Return user-friendly error to UI
        error_msg = f"❌ Pipeline Error: {str(e)}\n\nCheck console for full details."
        return None, error_msg

# Define the UI
# Define the UI
with gr.Blocks(title="Studio AI Dubber", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as app:
    
    with gr.Row():
        gr.Markdown(
            """
            # 🎙️ Studio-Level AI Video Dubber
            ### *Professional Voice Cloning, Translation, and Lip-Sync Pipeline*
            """
        )

    with gr.Row():
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Source Media")
                input_video = gr.File(label="Upload Video (MP4/MKV)", file_types=[".mp4", ".mkv", ".mov"])
            
            with gr.Group():
                gr.Markdown("### 2. Localization Settings")
                target_lang_dropdown = gr.Dropdown(
                    choices=[
                        ("Hindi (हिन्दी)", "hi"), 
                        ("Urdu (اردو)", "ur"), 
                        ("English", "en"), 
                        ("Spanish (Español)", "es"), 
                        ("French (Français)", "fr"), 
                        ("German (Deutsch)", "de"), 
                        ("Japanese (日本語)", "ja")
                    ], 
                    value="hi", 
                    label="Target Language"
                )
                lipsync_chk = gr.Checkbox(label="✅ Enable AI Lip-Sync (Wav2Lip)", value=True)
                bgm_chk = gr.Checkbox(label="🎵 Include Background Music", value=True)

            with gr.Accordion("⚙️ Advanced Audio Settings (Pro)", open=False):
                tone_dropdown = gr.Dropdown(
                    choices=["default", "happy", "sad", "angry", "surprised"],
                    value="default",
                    label="Emotion/Tone Preference (TTS)"
                )
                gr.Markdown("#### 🎤 RVC Voice Refining (Optional)")
                rvc_model_input = gr.File(
                    label="Upload Custom Voice Model (.pth)", 
                    file_types=[".pth"]
                )

            submit_btn = gr.Button("🚀 Start Studio Dubbing", variant="primary", size="lg")
        
        # RIGHT COLUMN: OUTPUTS
        with gr.Column(scale=1):
            gr.Markdown("### 3. Studio Output")
            output_video = gr.Video(label="Final Dubbed Video", interactive=False)
            status_msg = gr.Textbox(label="Pipeline Status", placeholder="Ready to dub...", interactive=False, lines=1)
            
            with gr.Accordion("🔍 Output Details", open=False):
                # Placeholder for extra logs or files if needed later
                gr.Markdown("*Detailed processing logs will appear in the console.*")

    # Footer
    with gr.Row():
        gr.Markdown("---")
        gr.Markdown("Build with ❤️ by Jarair Ahmad | Studio AI Dubber v1.0")

    submit_btn.click(
        fn=run_dubbing,
        inputs=[input_video, target_lang_dropdown, lipsync_chk, bgm_chk, tone_dropdown, rvc_model_input],
        outputs=[output_video, status_msg]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
