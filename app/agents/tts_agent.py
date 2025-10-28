"""
TTS (Text-to-Speech) Agent
Handles text-to-speech conversion and voice cloning using ElevenLabs API.
"""

import os
import time
import uuid
import mimetypes
import requests
from typing import Optional, Tuple


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




class TTSAgent:
    """Agent for text-to-speech using ElevenLabs API."""
    
    def __init__(self, api_key: str):
        """
        Initialize TTS agent.
        
        Args:
            api_key: ElevenLabs API key
        """
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
    

    def create_voice_clone(
        self,
        name: str,
        audio_path: str,
        description: str = "Cloned voice",
        timeout: int = 90
    ) -> Optional[str]:
        """
        Create a voice clone from reference audio.
        
        Args:
            name: Name for the cloned voice
            audio_path: Path to reference audio file
            description: Description of the voice
            timeout: Maximum time to wait for voice creation (seconds)
        
        Returns:
            Voice ID if successful, None otherwise
        """
        logger.info(f"Creating voice clone: {name}")
        logger.info(f"Reference audio: {audio_path}")
        
        # Infer MIME type
        mime_type, _ = mimetypes.guess_type(audio_path)
        if not mime_type:
            mime_type = "audio/mpeg"
        
        try:
            with open(audio_path, 'rb') as audio_file:
                files = [
                    ("files", (os.path.basename(audio_path), audio_file, mime_type))
                ]
                
                data = {
                    "name": name,
                    "description": description
                }
                
                headers = {"xi-api-key": self.api_key}
                
                response = requests.post(
                    f"{self.base_url}/voices/add",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code not in [200, 201]:
                    logger.error(f"Voice creation failed: {response.status_code} - {response.text}")
                    return None
                
                voice_data = response.json()
                voice_id = voice_data.get("voice_id")
                
                if not voice_id:
                    logger.error(f"No voice_id in response: {voice_data}")
                    return None
                
                logger.info(f"Voice clone created: {voice_id}")
                
                # Wait for voice to be ready
                time.sleep(1)
                if self._check_voice_ready(voice_id):
                    logger.info(f"Voice clone ready: {voice_id}")
                    return voice_id
                else:
                    logger.warning(f"Voice may not be fully ready: {voice_id}")
                    return voice_id
        
        except Exception as e:
            logger.error(f"Error creating voice clone: {str(e)}")
            return None
    

    def _check_voice_ready(self, voice_id: str, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """Check if voice is ready for use."""
        for i in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/voices/{voice_id}",
                    headers={"xi-api-key": self.api_key},
                    timeout=10
                )
                
                if response.status_code == 200:
                    voice_data = response.json()
                    samples = voice_data.get("samples", [])
                    if samples and len(samples) > 0:
                        return True
                
                if i < max_retries - 1:
                    time.sleep(retry_delay)
            
            except Exception as e:
                logger.warning(f"Error checking voice readiness: {str(e)}")
                if i < max_retries - 1:
                    time.sleep(retry_delay)
        
        return False
    

    def get_default_voice(self) -> str:
        """Get a default multilingual voice ID."""
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers={"xi-api-key": self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                voices_data = response.json()
                voices = voices_data.get("voices", [])
                
                # Look for multilingual voice
                for voice in voices:
                    labels = voice.get("labels", {})
                    if "use_case" in labels and "multilingual" in labels.get("use_case", "").lower():
                        voice_id = voice.get("voice_id")
                        logger.info(f"Using default multilingual voice: {voice_id}")
                        return voice_id
        
        except Exception as e:
            logger.warning(f"Error fetching default voice: {str(e)}")
        
        # Fallback to known voice ID (Adam)
        default_voice_id = "pNInz6obpgDQGcFmaJgB"
        logger.info(f"Using fallback voice: {default_voice_id}")
        return default_voice_id
    

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_192"
    ) -> Tuple[str, int]:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert
            output_path: Path to save audio file
            voice_id: ElevenLabs voice ID (uses default if None)
            model_id: TTS model to use
            output_format: Audio format
        
        Returns:
            Tuple of (output_path, audio_size_bytes)
        """
        if voice_id is None:
            voice_id = self.get_default_voice()
        
        logger.info(f"Generating TTS")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Voice ID: {voice_id}")
        logger.info(f"Model: {model_id}")
        
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "output_format": output_format
        }
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            error_detail = response.json() if response.content else response.text
            raise RuntimeError(f"ElevenLabs TTS error ({response.status_code}): {error_detail}")
        
        # Stream audio to file
        audio_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    audio_size += len(chunk)
        
        logger.info(f"Audio saved: {output_path} ({audio_size} bytes)")
        return output_path, audio_size
    
    
    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice.
        
        Args:
            voice_id: Voice ID to delete
        
        Returns:
            True if successful
        """
        try:
            response = requests.delete(
                f"{self.base_url}/voices/{voice_id}",
                headers={"xi-api-key": self.api_key},
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Voice deleted: {voice_id}")
                return True
            else:
                logger.warning(f"Voice deletion returned {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting voice: {str(e)}")
            return False

