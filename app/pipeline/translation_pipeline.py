"""
Video Translation Pipeline
Orchestrates the complete translation workflow using various agents.
"""

import os
import tempfile
import uuid
import json
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from pydub import AudioSegment
from tqdm import tqdm

from app.agents import ASRAgent, TranslationAgent, TTSAgent, SegmentationAgent, VideoAgent


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




class TranslationPipeline:
    """Orchestrates video translation with sentence-level segmentation."""
    
    def __init__(
        self,
        openai_api_key: str,
        elevenlabs_api_key: str
    ):
        """
        Initialize translation pipeline.
        
        Args:
            openai_api_key: OpenAI API key for ASR and translation
            elevenlabs_api_key: ElevenLabs API key for TTS
        """
        self.asr_agent = ASRAgent(openai_api_key)
        self.translation_agent = TranslationAgent(openai_api_key)
        self.tts_agent = TTSAgent(elevenlabs_api_key)
        self.segmentation_agent = SegmentationAgent(max_segment_duration_ms=15000)
        self.video_agent = VideoAgent()
    
    
    def translate_video(
        self,
        video_path: str,
        output_video_path: str,
        source_language: str = "English",
        target_language: str = "German",
        apply_leveling: bool = True,
        crossfade_ms: int = 20,
        save_intermediates: bool = True
    ) -> Tuple[str, int, Optional[str]]:
        """
        Translate video with sentence-level segmentation.
        
        Voice is automatically cloned from the video's audio track.
        
        Args:
            video_path: Path to input video
            output_video_path: Path to save output video
            source_language: Source language
            target_language: Target language
            apply_leveling: Whether to apply audio leveling
            crossfade_ms: Crossfade duration between segments
            save_intermediates: Save intermediate results to output directory
        
        Returns:
            Tuple of (output_video_path, number_of_segments, intermediates_dir)
        """
        logger.info("=" * 80)
        logger.info("VIDEO TRANSLATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input: {video_path}")
        logger.info(f"Languages: {source_language} → {target_language}")
        
        # Setup output directory structure
        video_filename = Path(video_path).stem
        intermediates_dir = None
        
        if save_intermediates:
            output_base = Path(output_video_path).parent
            intermediates_dir = output_base / video_filename
            intermediates_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving intermediates to: {intermediates_dir}")
        
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        # Track translations for saving
        segment_data = []
        
        try:
            # Step 1: Extract audio
            logger.info("\n[1/7] Extracting audio from video...")
            audio_path = os.path.join(temp_dir, "original_audio.mp3")
            self.video_agent.extract_audio(video_path, audio_path)
            temp_files.append(audio_path)
            
            # Save extracted English audio
            if save_intermediates:
                english_audio_path = intermediates_dir / f"{video_filename}_english_audio.mp3"
                import shutil
                shutil.copy2(audio_path, english_audio_path)
                logger.info(f"Saved English audio: {english_audio_path}")
            
            # Step 2: Transcribe with word timestamps
            logger.info("\n[2/7] Transcribing audio with word timestamps...")
            transcript_data = self.asr_agent.transcribe_with_timestamps(
                audio_path,
                language="en" if source_language == "English" else source_language.lower()[:2]
            )
            
            audio_duration_ms = int(self.asr_agent.get_audio_duration(audio_path) * 1000)
            
            # Save English transcript
            if save_intermediates:
                transcript_path = intermediates_dir / f"{video_filename}_english_transcript.json"
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved English transcript: {transcript_path}")
                
                # Save simple text version
                transcript_text_path = intermediates_dir / f"{video_filename}_english_transcript.txt"
                with open(transcript_text_path, 'w', encoding='utf-8') as f:
                    f.write(transcript_data.get('text', ''))
                logger.info(f"Saved English transcript (text): {transcript_text_path}")
            
            # Step 3: Segment by sentences
            logger.info("\n[3/7] Segmenting by sentences...")
            sentence_segments = self.segmentation_agent.segment_by_sentences(
                transcript_data,
                audio_duration_ms
            )
            
            if not sentence_segments:
                raise RuntimeError("No sentence segments found")
            
            logger.info(f"Detected {len(sentence_segments)} sentence segments")
            
            # Save English segments with timestamps
            if save_intermediates:
                segments_csv_path = intermediates_dir / f"{video_filename}_english_segments.csv"
                with open(segments_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['segment_id', 'start_ms', 'end_ms', 'duration_s', 'text'])
                    for idx, (start_ms, end_ms, text) in enumerate(sentence_segments):
                        duration_s = (end_ms - start_ms) / 1000.0
                        writer.writerow([idx + 1, start_ms, end_ms, f"{duration_s:.2f}", text])
                logger.info(f"Saved English segments: {segments_csv_path}")
            
            # Step 4: Clone voice from extracted audio
            logger.info("\n[4/7] Cloning voice from video audio...")
            voice_id = self.tts_agent.create_voice_clone(
                name=f"video_voice_{uuid.uuid4().hex[:8]}",
                audio_path=audio_path,
                description=f"Cloned voice from {Path(video_path).name}"
            )
            
            if not voice_id:
                logger.warning("Voice cloning failed, using default backup voice")
                voice_id = self.tts_agent.get_default_voice()
            else:
                logger.info(f"✓ Voice cloned successfully: {voice_id}")
            
            logger.info(f"Using voice ID: {voice_id}")
            
            # Step 5: Process segments (translate + TTS + retime)
            logger.info(f"\n[5/7] Processing {len(sentence_segments)} segments...")
            
            full_audio = AudioSegment.from_file(audio_path)
            retimed_segments = []
            
            progress_bar = tqdm(
                enumerate(sentence_segments),
                total=len(sentence_segments),
                desc="🎯 Processing segments",
                unit="segment",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                colour='blue'
            )
            
            for idx, (start_ms, end_ms, text) in progress_bar:
                progress_bar.set_description(f"🎯 Segment {idx+1}/{len(sentence_segments)}: {text[:25]}...")
                
                logger.debug(f"\n--- Segment {idx+1}/{len(sentence_segments)} ---")
                logger.debug(f"Time: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s")
                logger.debug(f"Text: {text[:60]}...")
                
                segment_duration_s = (end_ms - start_ms) / 1000.0
                
                # Translate
                translated_text = self.translation_agent.translate(
                    text,
                    source_language,
                    target_language,
                    time_budget_seconds=segment_duration_s
                )
                logger.debug(f"Translation: {translated_text[:60]}...")
                
                # Store segment data for saving
                segment_data.append({
                    'segment_id': idx + 1,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_s': segment_duration_s,
                    'english_text': text,
                    'german_text': translated_text
                })
                
                # TTS
                tts_path = os.path.join(temp_dir, f"seg_{idx:03d}_tts.mp3")
                self.tts_agent.text_to_speech(
                    translated_text,
                    tts_path,
                    voice_id=voice_id,
                    output_format="mp3_44100_192"
                )
                temp_files.append(tts_path)
                
                # Retime to match original duration
                retimed_path = os.path.join(temp_dir, f"seg_{idx:03d}_retimed.mp3")
                retimed_path, tempo_factor = self.video_agent.retime_audio(
                    tts_path,
                    segment_duration_s,
                    retimed_path,
                    tolerance_pct=0.03
                )
                temp_files.append(retimed_path)
                
                retimed_segments.append(retimed_path)
                logger.debug(f"Tempo factor: {tempo_factor:.4f}")
            
            progress_bar.close()
            
            # Save German translation and segments
            if save_intermediates:
                # Save German translation (full text)
                german_translation_path = intermediates_dir / f"{video_filename}_german_translation.txt"
                with open(german_translation_path, 'w', encoding='utf-8') as f:
                    for seg in segment_data:
                        f.write(seg['german_text'] + ' ')
                logger.info(f"Saved German translation: {german_translation_path}")
                
                # Save German segments with timestamps
                german_segments_csv_path = intermediates_dir / f"{video_filename}_german_segments.csv"
                with open(german_segments_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['segment_id', 'start_ms', 'end_ms', 'duration_s', 'english_text', 'german_text'])
                    for seg in segment_data:
                        writer.writerow([
                            seg['segment_id'],
                            seg['start_ms'],
                            seg['end_ms'],
                            f"{seg['duration_s']:.2f}",
                            seg['english_text'],
                            seg['german_text']
                        ])
                logger.info(f"Saved German segments: {german_segments_csv_path}")
                
                # Save JSON format for programmatic access
                segments_json_path = intermediates_dir / f"{video_filename}_segments.json"
                with open(segments_json_path, 'w', encoding='utf-8') as f:
                    json.dump(segment_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved segments (JSON): {segments_json_path}")
            
            # Step 6: Concatenate segments with timing preservation
            logger.info(f"\n[6/7] Concatenating {len(retimed_segments)} segments with timing...")
            concatenated_path = os.path.join(temp_dir, "concatenated.mp3")
            
            # Extract timing information from sentence_segments
            segment_timings = [(start_ms, end_ms) for start_ms, end_ms, _ in sentence_segments]
            
            self.video_agent.concatenate_audio_segments_with_timing(
                retimed_segments,
                segment_timings,
                concatenated_path,
                sample_rate=44100
            )
            temp_files.append(concatenated_path)
            
            # Apply leveling if requested
            final_audio_path = concatenated_path
            if apply_leveling:
                logger.info("Applying audio leveling...")
                leveled_path = os.path.join(temp_dir, "leveled.mp3")
                self.video_agent.apply_audio_leveling(
                    concatenated_path,
                    leveled_path,
                    target_loudness=-16.0
                )
                temp_files.append(leveled_path)
                final_audio_path = leveled_path
            
            # Save German audio
            if save_intermediates:
                german_audio_path = intermediates_dir / f"{video_filename}_german_audio.mp3"
                import shutil
                shutil.copy2(final_audio_path, german_audio_path)
                logger.info(f"Saved German audio: {german_audio_path}")
            
            # Step 7: Replace audio in video
            logger.info("\n[7/7] Creating final video with translated audio...")
            self.video_agent.replace_audio_in_video(
                video_path,
                final_audio_path,
                output_video_path,
                audio_codec="aac",
                audio_bitrate="192k"
            )
            
            # Save final video to intermediates directory
            if save_intermediates:
                final_video_path = intermediates_dir / f"{video_filename}_translated.mp4"
                import shutil
                shutil.copy2(output_video_path, final_video_path)
                logger.info(f"Saved final video: {final_video_path}")
            
            logger.info("\n" + "=" * 80)
            logger.info("TRANSLATION COMPLETE!")
            logger.info(f"Output: {output_video_path}")
            logger.info(f"Segments processed: {len(sentence_segments)}")
            if save_intermediates:
                logger.info(f"Intermediates saved to: {intermediates_dir}")
            logger.info("=" * 80)
            
            return output_video_path, len(sentence_segments), str(intermediates_dir) if intermediates_dir else None
        
        finally:
            # Cleanup temp files
            logger.debug("\nCleaning up temporary files...")
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not delete {temp_file}: {str(e)}")
            
            # Remove temp directory
            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass

