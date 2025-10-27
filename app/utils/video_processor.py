import os
import requests
import tempfile
import time
import uuid
import mimetypes
from pathlib import Path
from typing import Tuple, Optional
import ffmpeg
from openai import OpenAI


def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """
    Extract audio from video file and save as MP3.
    
    Args:
        video_path: Path to the input video file
        output_audio_path: Optional path for output audio file. If None, creates a temp file.
    
    Returns:
        Path to the extracted audio file
    """
    if output_audio_path is None:
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        output_audio_path = os.path.join(temp_dir, f"{Path(video_path).stem}_audio.mp3")
    
    try:
        # Extract audio using ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_audio_path, acodec='libmp3lame', audio_bitrate='192k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        return output_audio_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error extracting audio: {e.stderr.decode()}") from e


def transcribe_audio_with_whisper(
    audio_path: str,
    openai_api_key: str,
    language: str = "en",
    output_transcript_path: str = None
) -> Tuple[str, str]:
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_path: Path to the audio file
        openai_api_key: OpenAI API key
        language: Language code (default: "en" for English)
        output_transcript_path: Optional path to save transcript. If None, creates a temp file.
    
    Returns:
        Tuple of (transcript_text, transcript_file_path)
    """
    client = OpenAI(api_key=openai_api_key)
    
    try:
        # Open the audio file and send to Whisper API
        with open(audio_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )
        
        transcript_text = transcript_response
        
        # Save transcript to file
        if output_transcript_path is None:
            temp_dir = tempfile.gettempdir()
            output_transcript_path = os.path.join(
                temp_dir, 
                f"{Path(audio_path).stem}_transcript.txt"
            )
        
        with open(output_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        return transcript_text, output_transcript_path
    
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}") from e


def translate_text_with_gpt(
    text: str,
    openai_api_key: str,
    source_language: str = "English",
    target_language: str = "German",
    model: str = "gpt-4o-mini",
    output_translation_path: str = None
) -> Tuple[str, str]:
    """
    Translate text using OpenAI GPT model.
    
    Args:
        text: Text to translate
        openai_api_key: OpenAI API key
        source_language: Source language name (default: "English")
        target_language: Target language name (default: "German")
        model: GPT model to use (default: "gpt-4o-mini")
        output_translation_path: Optional path to save translation. If None, creates a temp file.
    
    Returns:
        Tuple of (translated_text, translation_file_path)
    """
    client = OpenAI(api_key=openai_api_key)
    
    try:
        # Create translation prompt
        system_prompt = (
            f"You are a professional translator. Translate the following text from "
            f"{source_language} to {target_language}. Maintain the original tone, style, "
            f"and meaning. Only provide the translation without any explanations or notes."
        )
        
        # Call GPT API for translation
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        # Save translation to file
        if output_translation_path is None:
            temp_dir = tempfile.gettempdir()
            output_translation_path = os.path.join(
                temp_dir,
                "translation.txt"
            )
        
        with open(output_translation_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        
        return translated_text, output_translation_path
    
    except Exception as e:
        raise RuntimeError(f"Error translating text: {str(e)}") from e


def create_voice_clone_elevenlabs(
    name: str,
    reference_audio_path: str,
    elevenlabs_api_key: str,
    description: str = "Cloned voice for video translation",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Create a voice clone in ElevenLabs using reference audio.
    
    Args:
        name: Name for the cloned voice
        reference_audio_path: Path to reference audio file
        elevenlabs_api_key: ElevenLabs API key
        description: Description for the voice
        max_retries: Maximum retries to wait for voice readiness
        retry_delay: Delay between retries in seconds
    
    Returns:
        voice_id of the created voice
    """
    try:
        url = "https://api.elevenlabs.io/v1/voices/add"
        
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        
        # Infer MIME type from file
        mime_type, _ = mimetypes.guess_type(reference_audio_path)
        if not mime_type or not mime_type.startswith('audio/'):
            mime_type = "audio/mpeg"  # fallback
        
        filename = os.path.basename(reference_audio_path)
        
        # Prepare multipart form data as list (supports multiple samples)
        with open(reference_audio_path, "rb") as audio_file:
            files = [
                ("files", (filename, audio_file, mime_type))
            ]
            
            data = {
                "name": name,
                "description": description
            }
            
            response = requests.post(
                url, 
                headers=headers, 
                files=files, 
                data=data,
                timeout=30
            )
        
        # Accept both 200 and 201 status codes
        if response.status_code not in [200, 201]:
            raise RuntimeError(
                f"ElevenLabs voice creation error: {response.status_code} - {response.text}"
            )
        
        response_data = response.json()
        voice_id = response_data.get("voice_id")
        
        if not voice_id:
            raise RuntimeError("No voice_id returned from ElevenLabs API")
        
        # Poll for voice readiness
        for attempt in range(max_retries):
            if _check_voice_ready(voice_id, elevenlabs_api_key):
                return voice_id
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        # Voice created but may not be fully ready; proceed anyway
        print(f"Warning: Voice {voice_id} may not be fully ready after {max_retries} retries")
        return voice_id
    
    except Exception as e:
        raise RuntimeError(f"Error creating voice clone: {str(e)}") from e


def _check_voice_ready(voice_id: str, elevenlabs_api_key: str) -> bool:
    """
    Check if a voice is ready for use.
    
    Args:
        voice_id: ID of the voice to check
        elevenlabs_api_key: ElevenLabs API key
    
    Returns:
        True if voice is ready, False otherwise
    """
    try:
        url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
        
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True
        
        return False
    
    except Exception:
        return False


def delete_voice_elevenlabs(voice_id: str, elevenlabs_api_key: str) -> None:
    """
    Delete a cloned voice from ElevenLabs.
    
    Args:
        voice_id: ID of the voice to delete
        elevenlabs_api_key: ElevenLabs API key
    """
    try:
        url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
        
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        
        response = requests.delete(url, headers=headers, timeout=10)
        
        # Accept 200, 201, or 204 as success
        if response.status_code not in [200, 201, 204]:
            print(f"Warning: Could not delete voice {voice_id}: {response.status_code}")
    
    except Exception as e:
        print(f"Warning: Error deleting voice {voice_id}: {str(e)}")


def text_to_speech_elevenlabs(
    text: str,
    elevenlabs_api_key: str,
    output_audio_path: str = None,
    voice_id: str = None,
    reference_audio_path: str = None,
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    use_speaker_boost: bool = True,
    output_format: str = "mp3_44100_128",
    cleanup_cloned_voice: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Convert text to speech using ElevenLabs API with optional voice cloning.
    
    Args:
        text: Text to convert to speech
        elevenlabs_api_key: ElevenLabs API key
        output_audio_path: Path to save the audio file. If None, generates unique filename.
        voice_id: ElevenLabs voice ID to use. If None and reference_audio provided, creates voice clone.
        reference_audio_path: Path to reference audio for voice cloning
        model_id: Model to use (eleven_multilingual_v2 supports multiple languages including German)
        stability: Voice stability (0.0 to 1.0)
        similarity_boost: Voice similarity boost (0.0 to 1.0)
        use_speaker_boost: Enable speaker boost for better quality
        output_format: Audio format (mp3_44100_128, mp3_44100_192, pcm_16000, pcm_22050, pcm_24000, pcm_44100)
        cleanup_cloned_voice: If True, deletes the cloned voice after synthesis (NOT recommended - expensive and rate-limited)
    
    Returns:
        Tuple of (audio_file_path, cloned_voice_id or None)
    """
    created_voice_id = None
    
    try:
        # Generate unique audio path if not provided
        if output_audio_path is None:
            temp_dir = tempfile.gettempdir()
            unique_id = uuid.uuid4().hex[:8]
            extension = "mp3" if output_format.startswith("mp3") else "pcm"
            output_audio_path = os.path.join(temp_dir, f"tts_{unique_id}.{extension}")
        
        # If voice cloning is requested with reference audio
        if reference_audio_path and not voice_id:
            # Create a voice clone (will be reusable)
            voice_name = f"clone_{uuid.uuid4().hex[:8]}"
            created_voice_id = create_voice_clone_elevenlabs(
                name=voice_name,
                reference_audio_path=reference_audio_path,
                elevenlabs_api_key=elevenlabs_api_key,
                description="Voice clone for video translation"
            )
            voice_id = created_voice_id
        
        # Use standard TTS with voice_id
        if not voice_id:
            # Use a default multilingual voice
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel - good for multiple languages
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "use_speaker_boost": use_speaker_boost
            },
            "output_format": output_format
        }
        
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            stream=True,
            timeout=60
        )
        
        # Check response
        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs TTS API error: {response.status_code} - {response.text}"
            )
        
        # Stream audio to file
        with open(output_audio_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Cleanup cloned voice if explicitly requested (NOT recommended)
        if created_voice_id and cleanup_cloned_voice:
            delete_voice_elevenlabs(created_voice_id, elevenlabs_api_key)
            return output_audio_path, None
        
        return output_audio_path, created_voice_id
    
    except Exception as e:
        # Cleanup on error only if cleanup was requested
        if created_voice_id and cleanup_cloned_voice:
            delete_voice_elevenlabs(created_voice_id, elevenlabs_api_key)
        raise RuntimeError(f"Error generating speech with ElevenLabs: {str(e)}") from e


def get_media_duration(file_path: str) -> float:
    """
    Get duration of media file (audio or video) in seconds.
    
    Args:
        file_path: Path to media file
    
    Returns:
        Duration in seconds
    """
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        raise RuntimeError(f"Error getting duration for {file_path}: {str(e)}") from e


def adjust_audio_duration(
    audio_path: str,
    target_duration: float,
    output_path: str = None
) -> str:
    """
    Adjust audio duration to match target duration.
    Pads with silence if too short, or cuts if too long.
    
    Args:
        audio_path: Path to input audio file
        target_duration: Target duration in seconds
        output_path: Path for adjusted audio. If None, creates temp file.
    
    Returns:
        Path to adjusted audio file
    """
    try:
        current_duration = get_media_duration(audio_path)
        
        # If durations match within 0.1 seconds, no adjustment needed
        if abs(current_duration - target_duration) < 0.1:
            return audio_path
        
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"adjusted_{uuid.uuid4().hex[:8]}.mp3")
        
        if current_duration < target_duration:
            # Audio is shorter - pad with silence at the end
            silence_duration = target_duration - current_duration
            stream = ffmpeg.input(audio_path)
            stream = ffmpeg.filter(stream, 'apad', pad_dur=silence_duration)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        else:
            # Audio is longer - trim to target duration
            stream = ffmpeg.input(audio_path, t=target_duration)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"Error adjusting audio duration: {str(e)}") from e


def replace_audio_in_video(
    video_path: str,
    new_audio_path: str,
    output_video_path: str = None,
    adjust_duration: bool = True,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k"
) -> str:
    """
    Replace audio track in video with new audio.
    
    Args:
        video_path: Path to original video file
        new_audio_path: Path to new audio file to replace with
        output_video_path: Path for output video. If None, creates temp file.
        adjust_duration: If True, adjusts audio duration to match video
        audio_codec: Audio codec to use (aac, mp3, etc.)
        audio_bitrate: Audio bitrate (e.g., '192k', '128k')
    
    Returns:
        Path to output video with replaced audio
    """
    adjusted_audio_path = None
    
    try:
        if output_video_path is None:
            temp_dir = tempfile.gettempdir()
            output_video_path = os.path.join(temp_dir, f"dubbed_{uuid.uuid4().hex[:8]}.mp4")
        
        # Adjust audio duration if requested
        audio_to_use = new_audio_path
        if adjust_duration:
            video_duration = get_media_duration(video_path)
            audio_to_use = adjust_audio_duration(new_audio_path, video_duration)
            if audio_to_use != new_audio_path:
                adjusted_audio_path = audio_to_use
        
        # Replace audio in video using FFmpeg
        # -map 0:v:0 = take video stream from input 0 (original video)
        # -map 1:a:0 = take audio stream from input 1 (new audio)
        # -c:v copy = copy video codec (no re-encoding)
        # -c:a aac = encode audio as AAC
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_to_use)
        
        output = ffmpeg.output(
            video_input['v'],
            audio_input['a'],
            output_video_path,
            vcodec='copy',  # Copy video without re-encoding
            acodec=audio_codec,
            audio_bitrate=audio_bitrate,
            shortest=None  # Don't cut based on shortest stream
        )
        
        ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        # Cleanup adjusted audio if it was created
        if adjusted_audio_path and adjusted_audio_path != new_audio_path:
            cleanup_temp_files(adjusted_audio_path)
        
        return output_video_path
    
    except ffmpeg.Error as e:
        # Cleanup on error
        if adjusted_audio_path and adjusted_audio_path != new_audio_path:
            cleanup_temp_files(adjusted_audio_path)
        raise RuntimeError(f"Error replacing audio in video: {e.stderr.decode() if e.stderr else str(e)}") from e
    
    except Exception as e:
        # Cleanup on error
        if adjusted_audio_path and adjusted_audio_path != new_audio_path:
            cleanup_temp_files(adjusted_audio_path)
        raise RuntimeError(f"Error replacing audio in video: {str(e)}") from e


def cleanup_temp_files(*file_paths: str) -> None:
    """
    Clean up temporary files.
    
    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {str(e)}")

