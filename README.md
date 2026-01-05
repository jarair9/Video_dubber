---
title: Video Dubber Studio
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Jarair Ahmad - AI Video Dubber (Studio Level)


**Jarair Ahmad** is a professional-grade AI video dubbing pipeline designed to provide high-quality, multi-speaker voice cloning and dubbing. It leverages state-of-the-art models for source separation, speaker diarization, and voice synthesis to create seamless localized videos.

## 🚀 Key Features

*   **Studio-Level Audio Separation**: Automatically splits vocals from background music (BGM) using `demucs`.
*   **Smart BGM Management**: Preserves the original background music and allows you to build a reusable BGM library in the `bgm/` folder.
*   **Speaker Diarization**: Identifies different speakers ("Character A", "Character B") and maintains consistent voice clones for each character throughout the video.
*   **Voice Cloning**: Uses `Chatterbox` and `OpenVoice` technology to clone voices from the best available reference clips.
*   **Multi-Language Support**: Translates and dubs content into Spanish, French, German, Italian, Hindi, and more.

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **FFmpeg**: Must be installed and added to your system PATH.
*   **Hugging Face Token**: Required for the Diarization model (`pyannote.audio`).
    *   Accept user conditions for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
    *   Set your token as an environment variable `HF_TOKEN` or login via CLI.

## 📦 Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-repo/ja-studio.git
    cd ja-studio
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter network errors with `huggingface_hub`, ensure you have installed the extra dependencies: `pip install huggingface_hub[hf_xet]`.*

## 🎬 Usage

### Basic Command
Run the main script with your input video and target language code:

```bash
python main.py video.mp4 --lang es
```

*   `video.mp4`: Path to your source video.
*   `--lang`: Target language code (e.g., `es` for Spanish, `fr` for French, `hi` for Hindi).

### Interactive Mode
If you run without arguments, it will prompt you for the language:
```bash
python main.py video.mp4
```

### Configuration
You can adjust default paths (Output folder, BGM folder) and model settings in `src/config.py`.

## 📂 Output Structure

*   **/outputs**: Contains the final dubbed video files.
*   **/bgm**: Stores extracted background music tracks (clean, without vocals).
*   **/output** (Temp): Stores intermediate files (separated stems, raw dubs) for debugging.

## ⚠️ Troubleshooting

*   **"No module named pyannote"**: Run `pip install pyannote.audio`.
*   **Hugging Face Login**: If Diarization fails, run `huggingface-cli login` and enter your read-access token.

---
**Powered by JA Studio**
