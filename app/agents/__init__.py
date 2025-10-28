"""
Agents for video translation pipeline.

Each agent handles a specific task:
- ASRAgent: Automatic Speech Recognition (Whisper API) / transcription
- TranslationAgent: Text translation (GPT API)
- TTSAgent: Text-to-Speech with voice cloning (ElevenLabs API)
- SegmentationAgent: Sentence-level segmentation
- VideoAgent: Video and audio processing (FFmpeg)
"""

from .asr_agent import ASRAgent
from .translation_agent import TranslationAgent
from .tts_agent import TTSAgent
from .segmentation_agent import SegmentationAgent
from .video_agent import VideoAgent

__all__ = [
    'ASRAgent',
    'TranslationAgent',
    'TTSAgent',
    'SegmentationAgent',
    'VideoAgent'
]

