"""
Translation Agent
Handles text translation with time-budget awareness using OpenAI GPT models.
"""

from typing import Optional
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




class TranslationAgent:
    """Agent for text translation using OpenAI GPT models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize translation agent.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        time_budget_seconds: Optional[float] = None
    ) -> str:
        """
        Translate text with optional time budget constraint.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            time_budget_seconds: Optional time constraint for translation
        
        Returns:
            Translated text
        """
        logger.debug(f"Translating: {source_language} → {target_language}")
        logger.debug(f"Text length: {len(text)} chars")
        
        # Build prompt with time budget if provided
        if time_budget_seconds:
            prompt = (
                f"Translate the following {source_language} text to {target_language}. "
                f"Keep the translation concise and natural, fitting within approximately "
                f"{time_budget_seconds:.1f} seconds of speech (±10%). "
                f"Maintain the original meaning while being brief.\n\n"
                f"Text: {text}"
            )
        else:
            prompt = (
                f"Translate the following {source_language} text to {target_language}. "
                f"Keep the translation natural and accurate.\n\n"
                f"Text: {text}"
            )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator specializing in {source_language} to {target_language} translation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        translated_text = response.choices[0].message.content.strip()
        logger.debug(f"Translation length: {len(translated_text)} chars")
        
        return translated_text

