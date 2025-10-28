"""
Segmentation Agent
Handles sentence-level segmentation using word timestamps from ASR.
"""

import re
from typing import List, Tuple, Dict


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




class SegmentationAgent:
    """Agent for sentence-level segmentation based on word timestamps."""
    
    def __init__(self, max_segment_duration_ms: int = 10000):
        """
        Initialize segmentation agent.
        
        Args:
            max_segment_duration_ms: Maximum duration for a single segment in milliseconds
        """
        self.max_segment_duration_ms = max_segment_duration_ms
    

    def segment_by_sentences(
        self,
        transcript_data: Dict,
        audio_duration_ms: int
    ) -> List[Tuple[int, int, str]]:
        """
        Segment audio by sentences using word timestamps.
        
        Args:
            transcript_data: Transcript data with word timestamps from ASR
            audio_duration_ms: Total audio duration in milliseconds
        
        Returns:
            List of (start_ms, end_ms, sentence_text) tuples
        """
        if not transcript_data or 'words' not in transcript_data:
            logger.warning("No word-level timestamps available")
            return []
        
        words = transcript_data['words']
        if not words:
            return []
        
        logger.info(f"Processing {len(words)} words for sentence boundaries")
        
        # Get full text for sentence detection
        full_text = transcript_data.get('text', '')
        
        # Split by sentence-ending punctuation
        sentence_endings = re.split(r'([.!?]+)\s+', full_text)
        
        # Reconstruct sentences (combining text + punctuation)
        raw_sentences = []
        i = 0
        while i < len(sentence_endings):
            if i + 1 < len(sentence_endings) and sentence_endings[i + 1].strip() in ['.', '!', '?', '..', '...']:
                raw_sentences.append(sentence_endings[i] + sentence_endings[i + 1])
                i += 2
            else:
                if sentence_endings[i].strip():
                    raw_sentences.append(sentence_endings[i])
                i += 1
        
        # If no sentences found, treat as single sentence
        if not raw_sentences:
            raw_sentences = [full_text]
        
        logger.info(f"Detected {len(raw_sentences)} sentences from transcript text")
        
        # Map sentences to word timestamps
        sentences = []
        word_idx = 0
        
        for sentence_text in raw_sentences:
            sentence_text = sentence_text.strip()
            if not sentence_text:
                continue
            
            # Find words belonging to this sentence
            sentence_words = []
            sentence_word_count = len(sentence_text.split())
            
            # Collect approximately the right number of words
            words_collected = 0
            while word_idx < len(words) and words_collected < sentence_word_count:
                sentence_words.append(words[word_idx])
                words_collected += 1
                word_idx += 1
            
            # If we have words for this sentence, create segment
            if sentence_words:
                start_ms = int(sentence_words[0].get('start', 0) * 1000)
                end_ms = int(sentence_words[-1].get('end', 0) * 1000)
                reconstructed_text = ' '.join([w.get('word', '').strip() for w in sentence_words])
                
                sentences.append((start_ms, end_ms, reconstructed_text))
        
        # Handle any remaining words
        if word_idx < len(words):
            remaining_words = words[word_idx:]
            start_ms = int(remaining_words[0].get('start', 0) * 1000)
            end_ms = int(remaining_words[-1].get('end', audio_duration_ms / 1000.0) * 1000)
            text = ' '.join([w.get('word', '').strip() for w in remaining_words])
            sentences.append((start_ms, end_ms, text))
        
        logger.info(f"Mapped sentences to {len(sentences)} time segments")
        
        # Split overly long sentences at phrase boundaries
        final_segments = self._split_long_sentences(sentences, words)
        
        logger.info(f"Final sentence-based segments: {len(final_segments)}")
        
        # Log sample segments
        for i, (s, e, txt) in enumerate(final_segments[:5]):
            logger.info(f"  Segment {i+1}: [{s/1000:.2f}s - {e/1000:.2f}s] ({(e-s)/1000:.2f}s) {txt[:60]}...")
        if len(final_segments) > 5:
            logger.info(f"  ... and {len(final_segments) - 5} more segments")
        
        return final_segments
    
    
    def _split_long_sentences(
        self,
        sentences: List[Tuple[int, int, str]],
        words: List[Dict]
    ) -> List[Tuple[int, int, str]]:
        """Split sentences that exceed max duration at phrase boundaries."""
        final_segments = []
        
        for start_ms, end_ms, text in sentences:
            duration_ms = end_ms - start_ms
            
            if duration_ms <= self.max_segment_duration_ms:
                final_segments.append((start_ms, end_ms, text))
            else:
                # Sentence is too long - split at phrase boundaries
                logger.info(f"Splitting long sentence ({duration_ms/1000:.1f}s): {text[:60]}...")
                
                # Find all words in this sentence
                sentence_word_objs = [w for w in words 
                                      if int(w.get('start', 0) * 1000) >= start_ms 
                                      and int(w.get('end', 0) * 1000) <= end_ms]
                
                if not sentence_word_objs:
                    final_segments.append((start_ms, end_ms, text))
                    continue
                
                # Split at natural phrase breaks
                phrase_words = []
                phrase_start_ms = None
                
                for word_obj in sentence_word_objs:
                    word_text = word_obj.get('word', '').strip()
                    word_start = int(word_obj.get('start', 0) * 1000)
                    word_end = int(word_obj.get('end', 0) * 1000)
                    
                    if phrase_start_ms is None:
                        phrase_start_ms = word_start
                    
                    phrase_words.append(word_obj)
                    current_duration = word_end - phrase_start_ms
                    
                    # Check for natural break points
                    is_comma = word_text.endswith(',')
                    is_conjunction = word_text.lower() in ['and', 'but', 'or', 'yet', 'so', 'because', 'while']
                    is_too_long = current_duration >= self.max_segment_duration_ms
                    is_good_length = current_duration >= self.max_segment_duration_ms * 0.4
                    
                    # Split if: too long OR (good length AND found natural break)
                    if is_too_long or (is_good_length and (is_comma or is_conjunction)):
                        if phrase_words:
                            phrase_end = int(phrase_words[-1].get('end', 0) * 1000)
                            phrase_text = ' '.join([w.get('word', '').strip() for w in phrase_words])
                            final_segments.append((phrase_start_ms, phrase_end, phrase_text))
                            
                            phrase_words = []
                            phrase_start_ms = None
                
                # Add remaining phrase
                if phrase_words:
                    phrase_start = int(phrase_words[0].get('start', 0) * 1000)
                    phrase_end = int(phrase_words[-1].get('end', 0) * 1000)
                    phrase_text = ' '.join([w.get('word', '').strip() for w in phrase_words])
                    final_segments.append((phrase_start, phrase_end, phrase_text))
        
        return final_segments

