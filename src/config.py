import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Base Paths
    BASE_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(BASE_DIR, "output") # Final video here
    TEMP_DIR = os.path.join(BASE_DIR, "temp") # Intermediate files here
    # Current config had TEMP_DIR=output. If I make OUTPUT_DIR=output, they overlap.
    # Let's make OUTPUT_DIR = output, and TEMP_DIR = output/temp to keep it clean, or just output.
    # User's request "make the output to temps @[output]" implies they want the result in 'output'.
    # If TEMP is also 'output', it's fine, just messy.
    # Let's set OUTPUT_DIR = "output" and TEMP_DIR = "output/temp" for cleanliness?
    # No, user has files in 'output' already.
    # I will set OUTPUT_DIR = "output" and TEMP_DIR = "output" (shared).
    # Actually, user said "save it in bgm folder" and "save in output". 
    # Let's interpret "save it in bgm folder" as a top level folder or inside output.
    # To be safe and organized:
    BGM_DIR = os.path.join(BASE_DIR, "bgm")
    
    # Model Configurations
    WHISPER_MODEL_SIZE = "base"
    WHISPER_MODEL_SIZE = "base"
    DEMUCS_MODEL = "mdx_extra_q" # 'htdemucs' (fast) < 'htdemucs_ft' < 'mdx_extra_q' (Best Vocal Isolation) 
    
    # Diarization
    # You might need to set your HF token as an env var: HF_TOKEN
    USE_DIARIZATION = True

    # Translation (LLM)
    TRANSLATION_SERVICE = os.getenv("TRANSLATION_SERVICE", "mistral") # 'openrouter', 'mistral', 'google'
    
    # OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = "z-ai/glm-4.5-air:free"
    
    # Mistral
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL = "mistral-small-latest" # Good balance of speed/quality
    
    
    # Dubbing Settings
    TARGET_LANGUAGE = "hi" # 'ur' = Urdu, 'hi' = Hindi, 'en' = English
    KEEP_BGM = True # Include background music by default
    
    
    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.BGM_DIR, exist_ok=True)
