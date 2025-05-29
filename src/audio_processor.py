"""
Audio processing module using faster-whisper for speech recognition.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tempfile
import subprocess

try:
    from faster_whisper import WhisperModel
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    
from loguru import logger


class AudioProcessor:
    """Handles audio processing and speech recognition using faster-whisper."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}
        
    async def initialize_model(self):
        """Initialize the Whisper model."""
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("Audio processing libraries not available. Using fallback mode.")
            return
            
        try:
            # Run model initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
            )
            logger.info(f"Whisper model '{self.model_size}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            logger.info("Continuing with fallback audio processing")
    
    def _validate_audio_file(self, file_path: Path) -> bool:
        """Validate audio file format and existence."""
        if not file_path.exists():
            logger.error(f"Audio file does not exist: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported audio format: {file_path.suffix}")
            return False
        
        return True
    
    def _get_audio_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract audio file metadata."""
        if not AUDIO_LIBS_AVAILABLE:
            return {
                'duration': 0,
                'sample_rate': 16000,
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'format': file_path.suffix.lower().lstrip('.'),
                'channels': 1
            }
            
        try:
            # Use librosa to get audio info
            y, sr = librosa.load(str(file_path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Detect format from file extension
            audio_format = file_path.suffix.lower().lstrip('.')
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'file_size': file_size,
                'format': audio_format,
                'channels': 1 if len(y.shape) == 1 else y.shape[0]
            }
        except Exception as e:
            logger.error(f"Failed to get audio metadata: {e}")
            return {}
    
    def _preprocess_audio(self, file_path: Path) -> Path:
        """Preprocess audio file for optimal recognition."""
        if not AUDIO_LIBS_AVAILABLE:
            return file_path
            
        try:
            # Load audio with pydub for format conversion
            audio = AudioSegment.from_file(str(file_path))
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Normalize audio level
            audio = audio.normalize()
            
            # Set sample rate to 16kHz (optimal for Whisper)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            
            # Export to temporary WAV file
            temp_file = Path(tempfile.mktemp(suffix='.wav'))
            audio.export(str(temp_file), format='wav')
            
            logger.info(f"Audio preprocessed: {file_path} -> {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return file_path  # Return original if preprocessing fails
    
    async def transcribe_audio(
        self, 
        file_path: Path, 
        language: Optional[str] = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text using faster-whisper.
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'ja', 'en'). None for auto-detection
            temperature: Sampling temperature
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling
        
        Returns:
            Dictionary containing transcription results
        """
        if not self.model:
            # Fallback response when model is not available
            return {
                'text': f'[Audio transcription unavailable] File: {file_path.name}',
                'language': language or 'unknown',
                'language_probability': 0.5,
                'duration': 0,
                'processing_time': 0.1,
                'model_used': 'fallback',
                'metadata': self._get_audio_metadata(file_path)
            }
        
        if not self._validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        start_time = time.time()
        
        try:
            # Get audio metadata
            metadata = self._get_audio_metadata(file_path)
            
            # Preprocess audio
            processed_file = self._preprocess_audio(file_path)
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    str(processed_file),
                    language=language,
                    temperature=temperature,
                    beam_size=beam_size,
                    best_of=best_of,
                    word_timestamps=True
                )
            )
            
            # Process segments
            transcribed_segments = []
            full_text = ""
            
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        segment_data['words'].append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })
                
                transcribed_segments.append(segment_data)
                full_text += segment.text.strip() + " "
            
            processing_time = time.time() - start_time
            
            # Clean up temporary file if created
            if processed_file != file_path and processed_file.exists():
                processed_file.unlink()
            
            result = {
                'text': full_text.strip(),
                'segments': transcribed_segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'processing_time': processing_time,
                'model_used': self.model_size,
                'metadata': metadata
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {e}")
            # Return fallback result on error
            return {
                'text': f'[Transcription error] {str(e)}',
                'language': language or 'unknown',
                'language_probability': 0.0,
                'duration': 0,
                'processing_time': time.time() - start_time,
                'model_used': self.model_size,
                'error': str(e),
                'metadata': self._get_audio_metadata(file_path)
            }
    
    async def transcribe_audio_stream(
        self,
        audio_data: bytes,
        format: str = 'wav',
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio from bytes stream.
        
        Args:
            audio_data: Audio data as bytes
            format: Audio format ('wav', 'mp3', etc.)
            language: Language code
        
        Returns:
            Dictionary containing transcription results
        """
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = Path(temp_file.name)
        
        try:
            result = await self.transcribe_audio(temp_path, language=language)
            return result
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    async def batch_transcribe(
        self,
        file_paths: list[Path],
        language: Optional[str] = None,
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Transcribe multiple audio files concurrently.
        
        Args:
            file_paths: List of audio file paths
            language: Language code
            max_concurrent: Maximum concurrent transcriptions
        
        Returns:
            Dictionary mapping file paths to transcription results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_single(file_path: Path):
            async with semaphore:
                try:
                    return str(file_path), await self.transcribe_audio(file_path, language=language)
                except Exception as e:
                    logger.error(f"Failed to transcribe {file_path}: {e}")
                    return str(file_path), {'error': str(e)}
        
        tasks = [transcribe_single(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'is_initialized': self.model is not None,
            'audio_libs_available': AUDIO_LIBS_AVAILABLE,
            'supported_formats': list(self.supported_formats)
        }
    
    async def change_model(self, model_size: str):
        """Change the Whisper model size."""
        if model_size != self.model_size:
            logger.info(f"Changing model from {self.model_size} to {model_size}")
            self.model_size = model_size
            self.model = None  # Clear current model
            await self.initialize_model()
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages for Whisper."""
        return {
            'ja': 'Japanese',
            'en': 'English', 
            'zh': 'Chinese',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese'
        }
    
    async def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed audio file information."""
        if not self._validate_audio_file(file_path):
            raise ValueError(f"Invalid audio file: {file_path}")
        
        metadata = self._get_audio_metadata(file_path)
        
        return {
            'filename': file_path.name,
            'size': metadata.get('file_size', 0),
            'duration': metadata.get('duration', 0),
            'format': metadata.get('format', ''),
            'sample_rate': metadata.get('sample_rate', 0),
            'channels': metadata.get('channels', 1),
            'supported': file_path.suffix.lower() in self.supported_formats
        }


# Global audio processor instance
audio_processor = AudioProcessor()