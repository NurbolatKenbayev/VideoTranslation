import os
import tempfile
from pathlib import Path
from typing import Tuple
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

