"""
Video Agent
Handles video and audio processing using FFmpeg.
"""

import os
import ffmpeg
from pydub import AudioSegment
from typing import Tuple


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




class VideoAgent:
    """Agent for video and audio processing using FFmpeg."""
    
    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio from video file as MP3.
        
        Args:
            video_path: Path to input video
            output_audio_path: Path to save extracted audio (MP3)
        
        Returns:
            Path to extracted audio
        """
        logger.info(f"Extracting audio from: {video_path}")
        
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_audio_path, acodec='libmp3lame', audio_bitrate='192k', ar=44100)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
            
            logger.info(f"Audio extracted: {output_audio_path}")
            return output_audio_path
        
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"Error extracting audio: {error_msg}") from e
    

    def get_duration(self, media_path: str) -> float:
        """
        Get media file duration in seconds.
        
        Args:
            media_path: Path to media file
        
        Returns:
            Duration in seconds
        """
        probe = ffmpeg.probe(media_path)
        duration = float(probe['format']['duration'])
        return duration
    

    def extract_audio_segment(
        self,
        full_audio: AudioSegment,
        start_ms: int,
        end_ms: int,
        output_path: str
    ) -> str:
        """
        Extract a segment from audio as MP3.
        
        Args:
            full_audio: Full audio as AudioSegment
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            output_path: Path to save segment (MP3)
        
        Returns:
            Path to extracted segment
        """
        segment = full_audio[start_ms:end_ms]
        segment.export(output_path, format="mp3", bitrate="192k")
        logger.debug(f"Audio segment extracted: {start_ms}ms-{end_ms}ms → {output_path}")
        return output_path
    
    
    def retime_audio(
        self,
        input_audio_path: str,
        target_duration_s: float,
        output_path: str,
        tolerance_pct: float = 0.03
    ) -> Tuple[str, float]:
        """
        Retime audio to match target duration using FFmpeg atempo filter.
        
        Args:
            input_audio_path: Path to input audio
            target_duration_s: Target duration in seconds
            output_path: Path to save retimed audio
            tolerance_pct: Tolerance percentage (0.03 = 3%)
        
        Returns:
            Tuple of (output_path, tempo_factor_applied)
        """
        # Get current duration
        current_duration = self.get_duration(input_audio_path)
        
        # Calculate tempo factor
        tempo_factor = current_duration / target_duration_s
        
        # Check if within tolerance
        if abs(tempo_factor - 1.0) < tolerance_pct:
            logger.debug(f"Duration difference within tolerance ({tolerance_pct*100}%), using original")
            # Just copy the file
            import shutil
            shutil.copy2(input_audio_path, output_path)
            return output_path, 1.0
        
        logger.debug(f"Retiming audio: {current_duration:.2f}s → {target_duration_s:.2f}s (tempo={tempo_factor:.4f})")
        
        # FFmpeg atempo filter has range 0.5-2.0
        # Chain multiple filters if needed
        stream = ffmpeg.input(input_audio_path)
        
        if 0.5 <= tempo_factor <= 2.0:
            # Single atempo filter
            stream = stream.filter('atempo', tempo_factor)
        else:
            # Chain multiple atempo filters
            remaining_factor = tempo_factor
            while remaining_factor > 2.0:
                stream = stream.filter('atempo', 2.0)
                remaining_factor /= 2.0
            while remaining_factor < 0.5:
                stream = stream.filter('atempo', 0.5)
                remaining_factor /= 0.5
            if abs(remaining_factor - 1.0) > 0.01:
                stream = stream.filter('atempo', remaining_factor)
        
        stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k', ar=44100)
        
        try:
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
            logger.debug(f"Audio retimed successfully: {output_path}")
            return output_path, tempo_factor
        
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg retiming error: {error_msg}")
            raise RuntimeError(f"Error retiming audio segment: {error_msg}") from e
    

    def concatenate_audio_segments_with_timing(
        self,
        segment_paths: list,
        segment_timings: list,
        output_path: str,
        sample_rate: int = 44100
    ) -> str:
        """
        Concatenate audio segments with proper timing (preserving gaps/silences).
        
        Args:
            segment_paths: List of audio segment paths
            segment_timings: List of (start_ms, end_ms) tuples for each segment
            output_path: Path to save concatenated audio (MP3)
            sample_rate: Sample rate for output
        
        Returns:
            Path to concatenated audio
        """
        logger.info(f"Concatenating {len(segment_paths)} audio segments with timing preservation")
        
        if not segment_paths:
            raise ValueError("No audio segments to concatenate")
        
        if len(segment_paths) == 1:
            logger.info("Only one segment")
            combined = AudioSegment.from_file(segment_paths[0])
            combined = combined.set_frame_rate(sample_rate)
            combined.export(output_path, format="mp3", bitrate="192k")
            return output_path
        
        # Start with silence if first segment doesn't start at 0
        first_start_ms = segment_timings[0][0]
        if first_start_ms > 0:
            combined = AudioSegment.silent(duration=first_start_ms, frame_rate=sample_rate)
            logger.info(f"Adding {first_start_ms}ms leading silence")
        else:
            combined = AudioSegment.empty()
        
        # Add each segment with proper timing
        for idx, (segment_path, (start_ms, end_ms)) in enumerate(zip(segment_paths, segment_timings)):
            segment = AudioSegment.from_file(segment_path)
            
            # Calculate expected position of this segment
            expected_start_ms = start_ms
            current_length_ms = len(combined)
            
            # Add silence gap if needed
            if current_length_ms < expected_start_ms:
                gap_ms = expected_start_ms - current_length_ms
                logger.info(f"Segment {idx+1}: Adding {gap_ms}ms silence gap")
                silence = AudioSegment.silent(duration=gap_ms, frame_rate=sample_rate)
                combined = combined + silence
            
            # Add the segment
            combined = combined + segment
            logger.debug(f"Segment {idx+1}: Added at {expected_start_ms}ms, duration={len(segment)}ms")
        
        # Export with specified sample rate as MP3
        combined = combined.set_frame_rate(sample_rate)
        combined.export(output_path, format="mp3", bitrate="192k")
        
        final_duration_ms = len(combined)
        logger.info(f"Audio concatenated with timing: {output_path} (duration: {final_duration_ms/1000:.2f}s)")
        return output_path
    

    def apply_audio_leveling(
        self,
        audio_path: str,
        output_path: str,
        target_loudness: float = -16.0
    ) -> str:
        """
        Apply loudness normalization to audio as MP3.
        
        Args:
            audio_path: Path to input audio
            output_path: Path to save normalized audio (MP3)
            target_loudness: Target loudness in LUFS (default: -16.0)
        
        Returns:
            Path to normalized audio
        """
        logger.info(f"Applying audio leveling: target={target_loudness} LUFS")
        
        try:
            stream = ffmpeg.input(audio_path)
            stream = stream.filter('loudnorm', I=target_loudness, LRA=11, TP=-1.5)
            stream = ffmpeg.output(stream, output_path, acodec='libmp3lame', audio_bitrate='192k', ar=44100)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
            
            logger.info(f"Audio leveling applied: {output_path}")
            return output_path
        
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg leveling error: {error_msg}")
            raise RuntimeError(f"Error applying audio leveling: {error_msg}") from e
    
    
    def replace_audio_in_video(
        self,
        video_path: str,
        new_audio_path: str,
        output_video_path: str,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k"
    ) -> str:
        """
        Replace audio track in video.
        
        Audio should already match video duration through timing-aware concatenation.
        If there's a small mismatch, pads with silence to ensure video isn't trimmed.
        
        Args:
            video_path: Path to input video
            new_audio_path: Path to new audio
            output_video_path: Path to save output video
            audio_codec: Audio codec to use
            audio_bitrate: Audio bitrate
        
        Returns:
            Path to output video
        """
        logger.info(f"Replacing audio in video: {video_path}")
        
        # Get durations for verification
        video_duration = self.get_duration(video_path)
        audio_duration = self.get_duration(new_audio_path)
        
        logger.info(f"Video duration: {video_duration:.2f}s")
        logger.info(f"Audio duration: {audio_duration:.2f}s")
        
        try:
            video_stream = ffmpeg.input(video_path).video
            audio_stream = ffmpeg.input(new_audio_path).audio
            
            # If audio is significantly shorter, pad it (safety measure)
            duration_diff = video_duration - audio_duration
            if duration_diff > 0.1:  # More than 100ms difference
                logger.warning(f"Audio is {duration_diff:.2f}s shorter than video, padding with silence")
                audio_stream = audio_stream.filter('apad', pad_dur=duration_diff)
            elif duration_diff < -0.1:  # Audio is longer
                logger.warning(f"Audio is {abs(duration_diff):.2f}s longer than video")
            else:
                logger.info("Audio and video durations match closely")
            
            stream = ffmpeg.output(
                video_stream,
                audio_stream,
                output_video_path,
                vcodec='copy',
                acodec=audio_codec,
                audio_bitrate=audio_bitrate
            )
            
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True, quiet=True)
            
            logger.info(f"Video with new audio created: {output_video_path}")
            return output_video_path
        
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise RuntimeError(f"Error replacing audio in video: {error_msg}") from e

