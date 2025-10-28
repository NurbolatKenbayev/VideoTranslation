"""
Video Translation API
FastAPI application for translating videos with sentence-level segmentation.
"""

import os
import tempfile
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.pipeline import TranslationPipeline


import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# Initialize FastAPI
app = FastAPI(
    title="Video Translation API",
    description="Translate videos with sentence-level segmentation and duration matching",
    version="2.0.0"
)


@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "service": "Video Translation API",
        "version": "2.0.0",
        "description": "Sentence-level video translation with audio synchronization",
        "endpoints": {
            "/translate-video": "POST - Translate video with sentence-based segmentation",
            "/health": "GET - Health check"
        },
        "features": [
            "Automatic voice cloning from video audio",
            "Sentence-level segmentation using Whisper word timestamps",
            "Time-budget-aware translation",
            "Audio retiming for perfect sync",
            "Loudness normalization",
            "Saves all intermediate results"
        ]
    }


@app.post("/translate-video")
async def translate_video(
    video: UploadFile = File(..., description="Input video file (MP4)"),
    source_language: str = Form("English", description="Source language"),
    target_language: str = Form("German", description="Target language"),
    apply_leveling: bool = Form(True, description="Apply loudness normalization"),
    crossfade_ms: int = Form(20, description="Crossfade between segments in ms"),
    filename: str = Form(None, description="Optional filename for output")
):
    """
    Translate video with sentence-level segmentation.
    
    The system automatically clones the voice from the video's audio track.
    
    Pipeline:
    1. Extract audio from video
    2. Transcribe with Whisper (word timestamps)
    3. Clone voice from extracted audio
    4. Segment by sentences
    5. Translate each sentence with time budget
    6. Generate TTS for each sentence using cloned voice
    7. Retime each segment to match original duration
    8. Concatenate segments with crossfade
    9. Apply loudness normalization
    10. Replace audio in video
    
    Args:
        video: Input video file (MP4)
        source_language: Source language (default: English)
        target_language: Target language (default: German)
        apply_leveling: Apply loudness normalization (default: True)
        crossfade_ms: Crossfade duration in milliseconds (default: 20)
        filename: Custom output filename (optional)
    
    Returns:
        JSON with output video path, intermediates directory, and processing stats
    """
    # Validate API keys
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured"
        )
    
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not configured"
        )
    
    temp_dir = tempfile.mkdtemp()
    video_path = None
    
    try:
        # Save uploaded video
        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Prepare output path
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        if filename:
            output_filename = f"{filename}.mp4"
        else:
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            original_name = Path(video.filename).stem
            output_filename = f"{original_name}_translated_{unique_id}.mp4"
        
        output_video_path = str(output_dir / output_filename)
        
        # Run translation pipeline
        logger.info("Starting translation pipeline...")
        
        pipeline = TranslationPipeline(
            openai_api_key=OPENAI_API_KEY,
            elevenlabs_api_key=ELEVENLABS_API_KEY
        )
        
        output_path, num_segments, intermediates_dir = pipeline.translate_video(
            video_path=video_path,
            output_video_path=output_video_path,
            source_language=source_language,
            target_language=target_language,
            apply_leveling=apply_leveling,
            crossfade_ms=crossfade_ms,
            save_intermediates=True
        )
        
        logger.info(f"Translation complete: {output_path}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Video translated successfully (voice auto-cloned from video)",
            "output_video": output_path,
            "intermediates_directory": intermediates_dir,
            "original_video": video.filename,
            "source_language": source_language,
            "target_language": target_language,
            "segments_processed": num_segments,
            "saved_files": {
                "english_audio": f"{intermediates_dir}/{Path(video.filename).stem}_english_audio.mp3" if intermediates_dir else None,
                "english_transcript_json": f"{intermediates_dir}/{Path(video.filename).stem}_english_transcript.json" if intermediates_dir else None,
                "english_transcript_txt": f"{intermediates_dir}/{Path(video.filename).stem}_english_transcript.txt" if intermediates_dir else None,
                "english_segments_csv": f"{intermediates_dir}/{Path(video.filename).stem}_english_segments.csv" if intermediates_dir else None,
                "german_translation_txt": f"{intermediates_dir}/{Path(video.filename).stem}_german_translation.txt" if intermediates_dir else None,
                "german_segments_csv": f"{intermediates_dir}/{Path(video.filename).stem}_german_segments.csv" if intermediates_dir else None,
                "segments_json": f"{intermediates_dir}/{Path(video.filename).stem}_segments.json" if intermediates_dir else None,
                "german_audio": f"{intermediates_dir}/{Path(video.filename).stem}_german_audio.mp3" if intermediates_dir else None,
                "final_video": f"{intermediates_dir}/{Path(video.filename).stem}_translated.mp4" if intermediates_dir else None
            }
        })
    
    except Exception as e:
        import traceback
        logger.error(f"Translation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
        
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    openai_configured = bool(OPENAI_API_KEY)
    elevenlabs_configured = bool(ELEVENLABS_API_KEY)
    
    return {
        "status": "healthy" if (openai_configured and elevenlabs_configured) else "degraded",
        "openai_configured": openai_configured,
        "elevenlabs_configured": elevenlabs_configured
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

