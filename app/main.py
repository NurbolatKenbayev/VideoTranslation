import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.utils.video_processor import (
    extract_audio_from_video,
    transcribe_audio_with_whisper,
    translate_text_with_gpt,
    cleanup_temp_files
)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



app = FastAPI(
    title="Video Translation API",
    description="API for video translation pipeline - extract audio and transcribe",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Video Translation API",
        "endpoints": {
            "/transcribe": "POST - Extract audio from video and transcribe using Whisper API",
            "/translate": "POST - Translate text from English to German using GPT"
        }
    }


@app.post("/transcribe")
async def transcribe_video(
    video: UploadFile = File(..., description="MP4 video file in English"),
    save_transcript: bool = Form(False, description="Whether to save transcript to disk"),
    save_audio: bool = Form(False, description="Whether to save extracted audio to disk")
):
    """
    Extract audio from video file and transcribe it using OpenAI Whisper API.
    
    Args:
        video: MP4 video file to process
        save_transcript: If True, saves transcript to disk (default: False)
        save_audio: If True, saves extracted audio to disk (default: False)
    
    Returns:
        JSON response with transcript text and optionally file paths
    """
    # Validate file type
    if not video.filename.endswith('.mp4'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload an MP4 video file."
        )
    
    # Get OpenAI API key from environment
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not found in environment variables"
        )
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    video_path = None
    audio_path = None
    transcript_path = None
    
    try:
        # Save uploaded video to temporary file
        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Extract audio from video
        audio_filename = f"{Path(video.filename).stem}_audio.mp3"
        if save_audio:
            # Save to app directory
            audio_output_dir = Path(__file__).parent.parent / "output"
            audio_output_dir.mkdir(exist_ok=True)
            audio_path = str(audio_output_dir / audio_filename)
        else:
            audio_path = os.path.join(temp_dir, audio_filename)
        
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe audio using Whisper API
        transcript_filename = f"{Path(video.filename).stem}_transcript.txt"
        if save_transcript:
            # Save to app directory
            transcript_output_dir = Path(__file__).parent.parent / "output"
            transcript_output_dir.mkdir(exist_ok=True)
            transcript_path = str(transcript_output_dir / transcript_filename)
        else:
            transcript_path = os.path.join(temp_dir, transcript_filename)
        
        transcript_text, transcript_path = transcribe_audio_with_whisper(
            audio_path=audio_path,
            openai_api_key=OPENAI_API_KEY,
            language="en",
            output_transcript_path=transcript_path
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "transcript": transcript_text,
            "video_filename": video.filename
        }
        
        if save_transcript:
            response_data["transcript_file"] = transcript_path
        
        if save_audio:
            response_data["audio_file"] = audio_path
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        cleanup_files = [video_path]
        if not save_audio and audio_path:
            cleanup_files.append(audio_path)
        if not save_transcript and transcript_path:
            cleanup_files.append(transcript_path)
        
        cleanup_temp_files(*cleanup_files)
        
        # Remove temp directory if empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


@app.post("/translate")
async def translate_text(
    text: str = Form(..., description="English text to translate"),
    target_language: str = Form("German", description="Target language for translation"),
    source_language: str = Form("English", description="Source language of the text"),
    model: str = Form("gpt-4o-mini", description="GPT model to use for translation"),
    save_translation: bool = Form(True, description="Whether to save translation to disk"),
    filename: str = Form(None, description="Optional filename for saved translation")
):
    """
    Translate text from English to German (or other languages) using OpenAI GPT.
    
    Args:
        text: Text to translate (in English by default)
        target_language: Target language for translation (default: German)
        source_language: Source language of the text (default: English)
        model: GPT model to use (default: gpt-4o-mini)
        save_translation: If True, saves translation to disk (default: True)
        filename: Optional custom filename for saved translation
    
    Returns:
        JSON response with translated text and optionally file path
    """
    # Get OpenAI API key from environment
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not found in environment variables"
        )
    
    translation_path = None
    
    try:
        # Prepare output path if saving
        if save_translation:
            translation_output_dir = Path(__file__).parent.parent / "output"
            translation_output_dir.mkdir(exist_ok=True)
            
            if filename:
                translation_filename = filename if filename.endswith('.txt') else f"{filename}.txt"
            else:
                translation_filename = f"translation_{source_language}_to_{target_language}.txt"
            
            translation_path = str(translation_output_dir / translation_filename)
        
        # Translate text using GPT
        translated_text, translation_path = translate_text_with_gpt(
            text=text,
            openai_api_key=OPENAI_API_KEY,
            source_language=source_language,
            target_language=target_language,
            model=model,
            output_translation_path=translation_path
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "model_used": model
        }
        
        if save_translation and translation_path:
            response_data["translation_file"] = translation_path
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error translating text: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file if not saving
        if not save_translation and translation_path:
            cleanup_temp_files(translation_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    openai_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy",
        "openai_configured": openai_key_configured
    }

