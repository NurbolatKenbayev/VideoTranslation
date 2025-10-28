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
    text_to_speech_elevenlabs,
    replace_audio_in_video,
    get_media_duration,
    cleanup_temp_files
)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")



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
            "/translate": "POST - Translate text from English to German using GPT",
            "/tts": "POST - Convert German text to speech with voice cloning using ElevenLabs",
            "/dub": "POST - Replace audio track in video with new audio (video dubbing)",
            "/voices": "GET - List available ElevenLabs voices",
            "/health": "GET - Health check"
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


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="German text to convert to speech"),
    reference_audio: UploadFile = File(None, description="Optional reference audio for voice cloning"),
    voice_id: str = Form(None, description="ElevenLabs voice ID (if not using voice cloning)"),
    model_id: str = Form("eleven_multilingual_v2", description="ElevenLabs model ID"),
    stability: float = Form(0.5, description="Voice stability (0.0-1.0)"),
    similarity_boost: float = Form(0.75, description="Voice similarity boost (0.0-1.0)"),
    use_speaker_boost: bool = Form(True, description="Enable speaker boost for better quality"),
    output_format: str = Form("mp3_44100_128", description="Audio format (mp3_44100_128, mp3_44100_192, etc.)"),
    save_audio: bool = Form(True, description="Whether to save audio to disk"),
    filename: str = Form(None, description="Optional filename for saved audio"),
    cleanup_cloned_voice: bool = Form(False, description="Delete cloned voice after use (NOT recommended)")
):
    """
    Convert German text to speech using ElevenLabs API with optional voice cloning.
    
    Args:
        text: German text to convert to speech
        reference_audio: Optional audio file to clone voice from (e.g., English audio)
        voice_id: ElevenLabs voice ID to use (if not cloning voice)
        model_id: Model to use (default: eleven_multilingual_v2 for German)
        stability: Voice stability setting (0.0 to 1.0)
        similarity_boost: Voice similarity boost (0.0 to 1.0)
        use_speaker_boost: Enable speaker boost for better quality
        output_format: Audio format (mp3_44100_128, mp3_44100_192, pcm_16000, etc.)
        save_audio: If True, saves audio to disk (default: True)
        filename: Optional custom filename for saved audio
        cleanup_cloned_voice: If True, deletes cloned voice after synthesis (expensive, not recommended)
    
    Returns:
        JSON response with audio file path and voice_id
    """
    # Get ElevenLabs API key from environment
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not found in environment variables"
        )
    
    temp_dir = tempfile.mkdtemp()
    reference_audio_path = None
    output_audio_path = None
    
    try:
        # Handle reference audio if provided
        if reference_audio:
            reference_audio_path = os.path.join(temp_dir, reference_audio.filename)
            with open(reference_audio_path, "wb") as f:
                content = await reference_audio.read()
                f.write(content)
        
        # Prepare output audio path
        if save_audio:
            audio_output_dir = Path(__file__).parent.parent / "output"
            audio_output_dir.mkdir(exist_ok=True)
            
            if filename:
                # Ensure correct extension based on format
                extension = "mp3" if output_format.startswith("mp3") else "pcm"
                if not filename.endswith(f".{extension}"):
                    audio_filename = f"{filename}.{extension}"
                else:
                    audio_filename = filename
            else:
                # Generate unique filename
                import uuid
                unique_id = uuid.uuid4().hex[:8]
                extension = "mp3" if output_format.startswith("mp3") else "pcm"
                audio_filename = f"german_audio_{unique_id}.{extension}"
            
            output_audio_path = str(audio_output_dir / audio_filename)
        else:
            output_audio_path = None  # Will be auto-generated with UUID
        
        # Generate speech with ElevenLabs
        output_audio_path, cloned_voice_id = text_to_speech_elevenlabs(
            text=text,
            elevenlabs_api_key=ELEVENLABS_API_KEY,
            output_audio_path=output_audio_path,
            voice_id=voice_id,
            reference_audio_path=reference_audio_path,
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
            use_speaker_boost=use_speaker_boost,
            output_format=output_format,
            cleanup_cloned_voice=cleanup_cloned_voice
        )
        
        # Prepare response
        response_data = {
            "status": "success",
            "text": text,
            "voice_cloning_used": bool(reference_audio_path),
            "voice_id_used": voice_id if voice_id else cloned_voice_id,
            "cloned_voice_id": cloned_voice_id,
            "model_used": model_id,
            "output_format": output_format,
            "cleanup_cloned_voice": cleanup_cloned_voice
        }
        
        if cloned_voice_id and not cleanup_cloned_voice:
            response_data["note"] = "Voice clone created and saved. Reuse this voice_id for future requests to save API calls."
        
        if save_audio:
            response_data["audio_file"] = output_audio_path
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        cleanup_files = []
        if reference_audio_path:
            cleanup_files.append(reference_audio_path)
        if not save_audio and output_audio_path:
            cleanup_files.append(output_audio_path)
        
        cleanup_temp_files(*cleanup_files)
        
        # Remove temp directory if empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


@app.post("/dub")
async def dub_video(
    video: UploadFile = File(..., description="Original video file (MP4)"),
    new_audio: UploadFile = File(..., description="New audio file to replace original audio"),
    adjust_duration: bool = Form(True, description="Auto-adjust audio duration to match video"),
    audio_codec: str = Form("aac", description="Audio codec (aac, mp3, libmp3lame)"),
    audio_bitrate: str = Form("192k", description="Audio bitrate (e.g., 192k, 128k, 256k)"),
    save_video: bool = Form(True, description="Whether to save dubbed video to disk"),
    filename: str = Form(None, description="Optional filename for saved video")
):
    """
    Replace audio track in video with new audio (video dubbing).
    
    This endpoint takes the original video and new audio, then:
    1. Optionally adjusts audio duration to match video (pads silence or trims)
    2. Replaces the audio track while keeping the video stream intact
    3. Returns the dubbed video
    
    Args:
        video: Original video file (e.g., English video)
        new_audio: New audio file (e.g., German dubbed audio)
        adjust_duration: If True, auto-adjusts audio to match video length
        audio_codec: Audio codec to use (aac recommended for MP4)
        audio_bitrate: Audio bitrate for encoding
        save_video: If True, saves dubbed video to disk
        filename: Optional custom filename for saved video
    
    Returns:
        JSON response with dubbed video file path and duration info
    """
    temp_dir = tempfile.mkdtemp()
    video_path = None
    audio_path = None
    output_video_path = None
    
    try:
        # Save uploaded video to temporary file
        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Save uploaded audio to temporary file
        audio_path = os.path.join(temp_dir, new_audio.filename)
        with open(audio_path, "wb") as f:
            content = await new_audio.read()
            f.write(content)
        
        # Get durations for info
        video_duration = get_media_duration(video_path)
        audio_duration = get_media_duration(audio_path)
        
        # Prepare output video path
        if save_video:
            video_output_dir = Path(__file__).parent.parent / "output"
            video_output_dir.mkdir(exist_ok=True)
            
            if filename:
                # Ensure .mp4 extension
                if not filename.endswith('.mp4'):
                    video_filename = f"{filename}.mp4"
                else:
                    video_filename = filename
            else:
                # Generate unique filename
                import uuid
                unique_id = uuid.uuid4().hex[:8]
                original_name = Path(video.filename).stem
                video_filename = f"{original_name}_dubbed_{unique_id}.mp4"
            
            output_video_path = str(video_output_dir / video_filename)
        else:
            output_video_path = None  # Will be auto-generated
        
        # Replace audio in video
        output_video_path = replace_audio_in_video(
            video_path=video_path,
            new_audio_path=audio_path,
            output_video_path=output_video_path,
            adjust_duration=adjust_duration,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate
        )
        
        # Get final output duration
        output_duration = get_media_duration(output_video_path)
        
        # Prepare response
        response_data = {
            "status": "success",
            "original_video": video.filename,
            "new_audio": new_audio.filename,
            "video_duration": round(video_duration, 2),
            "audio_duration": round(audio_duration, 2),
            "output_duration": round(output_duration, 2),
            "duration_adjusted": adjust_duration,
            "audio_codec": audio_codec,
            "audio_bitrate": audio_bitrate
        }
        
        if save_video:
            response_data["output_video"] = output_video_path
        
        # Add duration adjustment info
        if adjust_duration:
            duration_diff = audio_duration - video_duration
            if abs(duration_diff) > 0.1:
                if duration_diff < 0:
                    response_data["adjustment"] = f"Audio was {abs(duration_diff):.2f}s shorter - padded with silence"
                else:
                    response_data["adjustment"] = f"Audio was {duration_diff:.2f}s longer - trimmed to match video"
            else:
                response_data["adjustment"] = "No adjustment needed - durations matched"
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error dubbing video: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        cleanup_files = []
        if video_path:
            cleanup_files.append(video_path)
        if audio_path:
            cleanup_files.append(audio_path)
        if not save_video and output_video_path:
            cleanup_files.append(output_video_path)
        
        cleanup_temp_files(*cleanup_files)
        
        # Remove temp directory if empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


@app.get("/voices")
async def list_voices():
    """
    List available ElevenLabs voices in your account.
    
    Returns:
        JSON response with list of available voices
    """
    import requests
    
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY not found in environment variables"
        )
    
    try:
        url = "https://api.elevenlabs.io/v1/voices"
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs API error: {response.text}"
            )
        
        voices_data = response.json()
        voices = voices_data.get("voices", [])
        
        # Format the response
        formatted_voices = []
        for voice in voices:
            formatted_voices.append({
                "voice_id": voice.get("voice_id"),
                "name": voice.get("name"),
                "category": voice.get("category"),
                "labels": voice.get("labels", {}),
                "description": voice.get("description", ""),
                "preview_url": voice.get("preview_url", "")
            })
        
        return JSONResponse(content={
            "status": "success",
            "count": len(formatted_voices),
            "voices": formatted_voices
        })
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching voices: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    openai_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    elevenlabs_key_configured = bool(os.getenv("ELEVENLABS_API_KEY"))
    return {
        "status": "healthy",
        "openai_configured": openai_key_configured,
        "elevenlabs_configured": elevenlabs_key_configured
    }

