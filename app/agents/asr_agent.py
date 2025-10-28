"""
ASR (Automatic Speech Recognition) Agent
Handles audio transcription using OpenAI Whisper API with word-level timestamps.
"""

import os
from typing import Dict, List, Tuple
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




class ASRAgent:
    """Agent for automatic speech recognition using OpenAI Whisper API."""
    
    def __init__(self, api_key: str):
        """
        Initialize ASR agent.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: str = "en"
    ) -> Dict:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'de')
        
        Returns:
            Dictionary with transcript data including word timestamps
        """
        logger.info(f"Transcribing audio: {audio_path}")
        logger.info(f"Language: {language}")
        
        with open(audio_path, "rb") as audio_file:
            transcript_response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        
        # Convert to dict
        if hasattr(transcript_response, 'model_dump'):
            transcript_data = transcript_response.model_dump()
        else:
            transcript_data = transcript_response
        
        logger.info(f"Transcription complete: {len(transcript_data.get('text', ''))} characters")
        logger.info(f"Words with timestamps: {len(transcript_data.get('words', []))}")
        
        return transcript_data
    
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Duration in seconds
        """
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
        logger.debug(f"Audio duration: {duration:.2f}s")
        return duration

