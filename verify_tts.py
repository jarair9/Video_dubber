import sys

print("Verifying TTS installation...")

try:
    import TTS
    print("✅ 'TTS' package found.")
    
    try:
        from TTS.api import TTS
        print("✅ 'TTS.api' is accessible.")
        print("\nSUCCESS: Coqui TTS is installed and ready!")
        print("Note: The first time you run the pipeline, it will download the XTTS v2 model (~2-3GB).")
        
    except ImportError:
        print("❌ 'TTS' package found, but 'TTS.api' import failed. Version mismatch?")
        
except ImportError:
    print("❌ 'TTS' package NOT found.")
    print("Please ensure the installation completes successfully.")
