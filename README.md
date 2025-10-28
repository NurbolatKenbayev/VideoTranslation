# Video Translation Pipeline

A **video translation system** using orchestration over AI Agents for automatic speech recognition, translation, voice cloning, and audio synchronization.

## Features

- 🎤 **Automatic Voice Cloning** - Clones the speaker's voice from the video
- 📝 **Sentence-Level Segmentation** - Precise segmentation using Whisper word timestamps
- 🌍 **Time-Budget-Aware Translation** - Keeps translations concise to match original timing
- 🎵 **Audio Retiming** - Adjusts each segment to match original duration perfectly
- 🔊 **Loudness Normalization** - Professional audio quality (-16 LUFS)
- 📊 **Complete Intermediates** - Saves all intermediate results for debugging and analysis

**Future improvements:**
- 🫦 **LipSync** postprocessing


## Architecture

The system has modular OOP design with separate agents for each task, so that one can debug/enhance each agent independently and iteratively improve the video translation pipeline.

### Agents

- **ASRAgent** - Automatic Speech Recognition using OpenAI Whisper API
- **SegmentationAgent** - Sentence-level segmentation based on word timestamps
- **TranslationAgent** - Text translation using GPT-4o-mini with time budgets
- **TTSAgent** - Text-to-Speech and voice cloning using ElevenLabs API
- **VideoAgent** - Video/audio processing using FFmpeg
- (to be done) **LipSyncAgent** - With local Wav2Lip model or Sync Labs API

### Pipeline

**TranslationPipeline** orchestrates all agents:
1. Extract audio from video
2. Transcribe with Whisper (word timestamps)
3. Clone voice from extracted audio
4. Segment by sentences
5. Translate each sentence with time budget
6. Generate TTS for each sentence using cloned voice
7. Retime each segment to match original duration
8. Concatenate segments with proper timing (preserving gaps)
9. Apply loudness normalization
10. Replace audio in video

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg
- Docker (optional)

### Local Setup

1. **Install FFmpeg**:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:

Create a `.env` file and pass required API keys:
- `OPENAI_API_KEY` - For Whisper ASR and GPT translation
- `ELEVENLABS_API_KEY` - For voice cloning and TTS

### Docker Setup

1. **Build and run with Docker Compose** (Recommended):
```bash
docker-compose up --build -d
```

2. **Or build and run manually**:
```bash
# Build (will include .env file in the image)
docker build -t video-translation .

# Run
docker run -d \
  --name video-translation-api \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/test_data:/app/test_data:ro \
  video-translation
```


## Usage

### Start the Server

**Local**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Docker**:
```bash
docker-compose up --build -d
```

### API Endpoints

#### Translate Video
```bash
curl -X POST "http://localhost:8000/translate-video" \
  -F "video=@input.mp4" \
  -F "source_language=English" \
  -F "target_language=German"
```

**Parameters**:
- `video` (required) - Input video file (MP4)
- `source_language` (optional, default: "English") - Source language
- `target_language` (optional, default: "German") - Target language
- `apply_leveling` (optional, default: true) - Apply loudness normalization
- `crossfade_ms` (optional, default: 20) - Crossfade between segments in ms
- `filename` (optional) - Custom output filename

#### Health Check
```bash
curl http://localhost:8000/health
```

### Output

All results are saved to `output/{video_filename}/`:
- `{name}_english_audio.mp3` - Extracted English audio
- `{name}_english_transcript.json` - Full Whisper transcript with word timestamps
- `{name}_english_transcript.txt` - Simple text transcript
- `{name}_english_segments.csv` - English segments with timestamps
- `{name}_german_translation.txt` - Full German translation
- `{name}_german_segments.csv` - German segments with timestamps (includes English comparison)
- `{name}_segments.json` - Combined segments in JSON format
- `{name}_german_audio.mp3` - Final German audio
- `{name}_translated.mp4` - Final translated video

## API Response Example

```json
{
  "status": "success",
  "message": "Video translated successfully (voice auto-cloned from video)",
  "output_video": "/path/to/output/video_translated.mp4",
  "intermediates_directory": "/path/to/output/video_name",
  "segments_processed": 10,
  "saved_files": {
    "english_audio": "...",
    "english_transcript_json": "...",
    "german_audio": "...",
    "final_video": "..."
  }
}
```

## Development

### Project Structure

```
VideoTranslation/
├── app/
│   ├── agents/           # Individual agent classes
│   │   ├── asr_agent.py
│   │   ├── segmentation_agent.py
│   │   ├── translation_agent.py
│   │   ├── tts_agent.py
│   │   └── video_agent.py
│   ├── pipeline/         # Pipeline orchestration
│   │   └── translation_pipeline.py
│   └── main.py           # FastAPI application
├── output/               # Generated files
├── test_data/            # Test videos
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Testing

```bash
# Test with sample video
curl -X POST "http://localhost:8000/translate-video" \
  -F "video=@test_data/Tanzania-2.mp4" \
  -F "source_language=English" \
  -F "target_language=German"
```

## Technical Details

### Audio Format
- All intermediate audio: MP3, 192kbps, 44.1kHz
- Final video audio: AAC, 192kbps, 44.1kHz

### Timing Preservation
- Each segment is retimed to match its original duration exactly
- Silence gaps between segments are preserved
- Output video duration = Input video duration

### Voice Cloning
- Automatically clones voice from video's audio track
- Fallback to default multilingual voice if cloning fails
- 90-second timeout with readiness checks


## Support

For issues and questions, please open an issue on GitHub.

