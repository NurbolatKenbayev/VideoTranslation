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



def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """
    Extract audio from video file and save as MP3.
    
    Args:
        video_path: Path to the input video file
        output_audio_path: Optional path for output audio file. If None, creates a temp file.
    
    Returns:
        Path to the extracted audio file
    """
    logger.info(f"Starting audio extraction from video: {video_path}")
    
    if output_audio_path is None:
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        output_audio_path = os.path.join(temp_dir, f"{Path(video_path).stem}_audio.mp3")
        logger.debug(f"No output path provided, using temp file: {output_audio_path}")
    
    try:
        # Extract audio using ffmpeg
        logger.info("Running ffmpeg to extract audio...")
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_audio_path, acodec='libmp3lame', audio_bitrate='192k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        logger.info(f"Audio extraction completed successfully: {output_audio_path}")
        return output_audio_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error during audio extraction: {e.stderr.decode()}")
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
    logger.info(f"Starting transcription of audio file: {audio_path}")
    logger.info(f"Transcription language: {language}")
    
    client = OpenAI(api_key=openai_api_key)
    
    try:
        # Open the audio file and send to Whisper API
        logger.info("Sending audio to OpenAI Whisper API...")
        with open(audio_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )
        
        transcript_text = transcript_response
        logger.info(f"Transcription received ({len(transcript_text)} characters)")
        logger.debug(f"Transcript preview: {transcript_text[:100]}...")
        
        # Save transcript to file
        if output_transcript_path is None:
            temp_dir = tempfile.gettempdir()
            output_transcript_path = os.path.join(
                temp_dir, 
                f"{Path(audio_path).stem}_transcript.txt"
            )
        
        with open(output_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        
        logger.info(f"Transcript saved to: {output_transcript_path}")
        return transcript_text, output_transcript_path
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
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
    logger.info(f"Starting translation: {source_language} -> {target_language}")
    logger.info(f"Using model: {model}")
    logger.info(f"Text length: {len(text)} characters")
    
    client = OpenAI(api_key=openai_api_key)
    
    try:
        # Create translation prompt
        system_prompt = (
            f"You are a professional translator. Translate the following text from "
            f"{source_language} to {target_language}. Maintain the original tone, style, "
            f"and meaning. Only provide the translation without any explanations or notes."
        )
        
        # Call GPT API for translation
        logger.info("Sending translation request to OpenAI GPT...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3  # Lower temperature for more consistent translations
        )
        
        translated_text = response.choices[0].message.content.strip()
        logger.info(f"Translation received ({len(translated_text)} characters)")
        logger.debug(f"Translation preview: {translated_text[:100]}...")
        
        # Save translation to file
        if output_translation_path is None:
            temp_dir = tempfile.gettempdir()
            output_translation_path = os.path.join(
                temp_dir,
                "translation.txt"
            )
        
        with open(output_translation_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        
        logger.info(f"Translation saved to: {output_translation_path}")
        return translated_text, output_translation_path
    
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise RuntimeError(f"Error translating text: {str(e)}") from e


def create_voice_clone_elevenlabs(
    name: str,
    reference_audio_path: str,
    elevenlabs_api_key: str,
    description: str = "Cloned voice for video translation",
    timeout: int = 90
) -> Optional[str]:
    """
    Create a voice clone in ElevenLabs using reference audio (synchronous).
    Returns voice_id if successful, None if timeout (caller should use backup voice).
    
    Args:
        name: Name for the cloned voice
        reference_audio_path: Path to reference audio file
        elevenlabs_api_key: ElevenLabs API key
        description: Description for the voice
        timeout: Timeout for synchronous voice creation in seconds (default: 90)
                 Sufficient for up to 2-minute reference audio files
    
    Returns:
        voice_id of the created voice, or None if timeout/failure
    """
    logger.info(f"Creating voice clone: {name}")
    logger.info(f"Reference audio: {reference_audio_path}")
    logger.info(f"Synchronous creation with {timeout}s timeout...")
    
    try:
        url = "https://api.elevenlabs.io/v1/voices/add"
        
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        
        # Infer MIME type from file
        mime_type, _ = mimetypes.guess_type(reference_audio_path)
        if not mime_type or not mime_type.startswith('audio/'):
            mime_type = "audio/mpeg"
        
        logger.debug(f"Detected MIME type: {mime_type}")
        filename = os.path.basename(reference_audio_path)
        
        # Upload with extended timeout to allow processing
        with open(reference_audio_path, "rb") as audio_file:
            files = [
                ("files", (filename, audio_file, mime_type))
            ]
            
            data = {
                "name": name,
                "description": description
            }
            
            logger.debug(f"Uploading audio for voice cloning...")
            response = requests.post(
                url, 
                headers=headers, 
                files=files, 
                data=data,
                timeout=timeout
            )
        
        # Check response
        if response.status_code not in [200, 201]:
            logger.error(f"Voice creation failed: {response.status_code} - {response.text}")
            logger.warning("Will use backup default voice")
            return None
        
        response_data = response.json()
        voice_id = response_data.get("voice_id")
        
        if not voice_id:
            logger.error("No voice_id returned from ElevenLabs API")
            logger.warning("Will use backup default voice")
            return None
        
        logger.info(f"Voice clone created with ID: {voice_id}")
        
        # Brief wait then verify voice is ready
        logger.debug("Verifying voice readiness...")
        time.sleep(1)
        
        if _check_voice_ready(voice_id, elevenlabs_api_key):
            logger.info(f"✓ Voice clone ready and verified: {voice_id}")
            return voice_id
        else:
            logger.warning("Voice created but readiness check failed")
            logger.info("Giving it one more moment...")
            time.sleep(3)
            
            if _check_voice_ready(voice_id, elevenlabs_api_key):
                logger.info(f"✓ Voice clone ready after additional wait: {voice_id}")
                return voice_id
            else:
                logger.warning(f"Voice {voice_id} not ready after verification attempts")
                logger.warning("Will use backup default voice")
                return None
    
    except requests.exceptions.Timeout:
        logger.warning(f"Voice creation timed out after {timeout}s")
        logger.info("Will use backup default voice instead")
        return None
    
    except Exception as e:
        logger.error(f"Error creating voice clone: {str(e)}")
        logger.warning("Will use backup default voice")
        return None


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
            # Voice exists and is accessible
            voice_data = response.json()
            
            # Check if voice has any samples (indicates it's fully processed)
            samples = voice_data.get("samples", [])
            if samples:
                logger.debug(f"Voice has {len(samples)} sample(s), voice is ready")
                return True
            else:
                logger.debug("Voice exists but no samples yet, still processing...")
                return False
        
        logger.debug(f"Voice check returned status {response.status_code}")
        return False
    
    except Exception as e:
        logger.debug(f"Error checking voice readiness: {str(e)}")
        return False


def _get_default_voice(elevenlabs_api_key: str) -> str:
    """
    Get a default voice ID from ElevenLabs available voices.
    
    Args:
        elevenlabs_api_key: ElevenLabs API key
    
    Returns:
        Voice ID string
    """
    logger.info("Fetching available voices from ElevenLabs...")
    
    try:
        url = "https://api.elevenlabs.io/v1/voices"
        
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            logger.info(f"Found {len(voices)} available voices")
            
            # Try to find a multilingual voice
            for voice in voices:
                labels = voice.get("labels", {})
                voice_name = voice.get("name", "Unknown")
                # Look for multilingual voices
                if "multilingual" in str(labels).lower():
                    voice_id = voice.get("voice_id")
                    logger.info(f"Selected multilingual voice: {voice_name} (ID: {voice_id})")
                    return voice_id
            
            # If no multilingual, return the first available voice
            if voices:
                voice_id = voices[0].get("voice_id")
                voice_name = voices[0].get("name", "Unknown")
                logger.info(f"No multilingual voice found, using first available: {voice_name} (ID: {voice_id})")
                return voice_id
        
        # Fallback to a commonly available voice ID
        logger.warning("Could not fetch voices from API, using fallback voice ID (Adam)")
        return "pNInz6obpgDQGcFmaJgB"
    
    except Exception as e:
        # If all else fails, use Adam's voice ID as fallback
        logger.error(f"Error fetching voices: {str(e)}, using fallback voice ID")
        return "pNInz6obpgDQGcFmaJgB"


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
    logger.info("Starting text-to-speech conversion with ElevenLabs")
    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Model: {model_id}, Format: {output_format}")
    
    created_voice_id = None
    
    try:
        # Generate unique audio path if not provided
        if output_audio_path is None:
            temp_dir = tempfile.gettempdir()
            unique_id = uuid.uuid4().hex[:8]
            extension = "mp3" if output_format.startswith("mp3") else "pcm"
            output_audio_path = os.path.join(temp_dir, f"tts_{unique_id}.{extension}")
            logger.debug(f"No output path provided, using: {output_audio_path}")
        
        # If voice cloning is requested with reference audio
        if reference_audio_path and not voice_id:
            logger.info("Voice cloning requested with reference audio")
            # Create a voice clone (will be reusable)
            voice_name = f"clone_{uuid.uuid4().hex[:8]}"
            created_voice_id = create_voice_clone_elevenlabs(
                name=voice_name,
                reference_audio_path=reference_audio_path,
                elevenlabs_api_key=elevenlabs_api_key,
                description="Voice clone for video translation"
            )
            
            if created_voice_id:
                voice_id = created_voice_id
                logger.info("Using cloned voice for TTS")
            else:
                logger.warning("Voice cloning failed, falling back to default voice")
                # voice_id remains None, will get default voice below
        
        # Use standard TTS with voice_id
        if not voice_id:
            logger.info("No voice_id provided, getting default voice...")
            # Get a default multilingual voice from available voices
            voice_id = _get_default_voice(elevenlabs_api_key)
        else:
            logger.info(f"Using provided voice_id: {voice_id}")
        
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
        
        logger.info("Sending TTS request to ElevenLabs...")
        logger.debug(f"Settings - Stability: {stability}, Similarity: {similarity_boost}, Speaker boost: {use_speaker_boost}")
        
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            stream=True,
            timeout=60
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"TTS API error: {response.status_code} - {response.text}")
            raise RuntimeError(
                f"ElevenLabs TTS API error: {response.status_code} - {response.text}"
            )
        
        logger.info("Streaming audio response to file...")
        # Stream audio to file
        bytes_written = 0
        with open(output_audio_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
        
        logger.info(f"Audio saved successfully ({bytes_written} bytes): {output_audio_path}")
        
        # Cleanup cloned voice if explicitly requested (NOT recommended)
        if created_voice_id and cleanup_cloned_voice:
            logger.info(f"Cleaning up cloned voice: {created_voice_id}")
            delete_voice_elevenlabs(created_voice_id, elevenlabs_api_key)
            return output_audio_path, None
        
        if created_voice_id:
            logger.info(f"Voice clone {created_voice_id} will be kept for reuse")
        
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
        logger.debug(f"Probing media duration: {file_path}")
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        logger.debug(f"Media duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        logger.error(f"Error probing media duration: {str(e)}")
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
    logger.info(f"Adjusting audio duration to match target: {target_duration:.2f}s")
    
    try:
        current_duration = get_media_duration(audio_path)
        duration_diff = current_duration - target_duration
        
        # If durations match within 0.1 seconds, no adjustment needed
        if abs(duration_diff) < 0.1:
            logger.info("Audio duration matches target (within 0.1s), no adjustment needed")
            return audio_path
        
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"adjusted_{uuid.uuid4().hex[:8]}.mp3")
        
        if current_duration < target_duration:
            # Audio is shorter - pad with silence at the end
            silence_duration = target_duration - current_duration
            logger.info(f"Audio is {silence_duration:.2f}s shorter, padding with silence")
            stream = ffmpeg.input(audio_path)
            stream = ffmpeg.filter(stream, 'apad', pad_dur=silence_duration)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        else:
            # Audio is longer - trim to target duration
            logger.info(f"Audio is {abs(duration_diff):.2f}s longer, trimming to match")
            stream = ffmpeg.input(audio_path, t=target_duration)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        logger.info(f"Audio adjustment completed: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error adjusting audio duration: {str(e)}")
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
    logger.info("Starting video dubbing (audio replacement)")
    logger.info(f"Video: {video_path}")
    logger.info(f"New audio: {new_audio_path}")
    logger.info(f"Audio codec: {audio_codec}, bitrate: {audio_bitrate}")
    
    adjusted_audio_path = None
    
    try:
        if output_video_path is None:
            temp_dir = tempfile.gettempdir()
            output_video_path = os.path.join(temp_dir, f"dubbed_{uuid.uuid4().hex[:8]}.mp4")
            logger.debug(f"No output path provided, using: {output_video_path}")
        
        # Adjust audio duration if requested
        audio_to_use = new_audio_path
        if adjust_duration:
            logger.info("Duration adjustment enabled")
            video_duration = get_media_duration(video_path)
            audio_to_use = adjust_audio_duration(new_audio_path, video_duration)
            if audio_to_use != new_audio_path:
                adjusted_audio_path = audio_to_use
                logger.info("Using adjusted audio for dubbing")
        else:
            logger.info("Duration adjustment disabled, using original audio")
        
        # Replace audio in video using FFmpeg
        logger.info("Running FFmpeg to replace audio track...")
        logger.debug("FFmpeg operation: copy video stream, encode new audio stream")
        
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
        
        logger.info(f"Video dubbing completed successfully: {output_video_path}")
        
        # Cleanup adjusted audio if it was created
        if adjusted_audio_path and adjusted_audio_path != new_audio_path:
            logger.debug(f"Cleaning up temporary adjusted audio: {adjusted_audio_path}")
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

