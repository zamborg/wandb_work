#!/usr/bin/env python3
"""
Audio Diarization Script using Whisper and pyannote-audio

This script takes an M4A audio file, performs speaker diarization,
transcribes the audio using Whisper, and outputs timestamped segments
with speaker labels.
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import whisper
    import torch
    import torchaudio
    from pydub import AudioSegment
    from pyannote.audio import Pipeline
    from pyannote.core import Interval, Segment
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


class AudioDiarizer:
    def __init__(self, whisper_model: str = "base", device: str = None):
        """
        Initialize the AudioDiarizer with Whisper and diarization models.
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Whisper model
        print(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        # Load diarization pipeline
        print("Loading diarization pipeline...")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # You may need to set up HuggingFace token for some models
            )
        except Exception as e:
            print(f"Error loading diarization pipeline: {e}")
            print("You may need to accept the terms and set up a HuggingFace token")
            print("Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
            sys.exit(1)
    
    def convert_m4a_to_wav(self, m4a_path: str, output_path: str = None) -> str:
        """
        Convert M4A file to WAV format for processing.
        
        Args:
            m4a_path: Path to the M4A file
            output_path: Optional output path for WAV file
            
        Returns:
            Path to the converted WAV file
        """
        if output_path is None:
            output_path = m4a_path.replace('.m4a', '_converted.wav')
        
        print(f"Converting {m4a_path} to WAV format...")
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio = audio.set_frame_rate(16000).set_channels(1)  # Optimize for speech processing
        audio.export(output_path, format="wav")
        print(f"Converted audio saved to: {output_path}")
        return output_path
    
    def perform_diarization(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing diarization results
        """
        print("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        
        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start
            })
        
        print(f"Found {len(speakers)} speakers")
        return speakers, diarization
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Whisper transcription result
        """
        print("Transcribing audio with Whisper...")
        result = whisper.transcribe(self.whisper_model, audio_path, word_timestamps=True)
        return result
    
    def align_transcription_with_speakers(self, transcription: Dict, diarization) -> List[Dict]:
        """
        Align Whisper transcription with speaker diarization results.
        
        Args:
            transcription: Whisper transcription result
            diarization: pyannote diarization result
            
        Returns:
            List of aligned segments with speaker and text
        """
        print("Aligning transcription with speaker diarization...")
        aligned_segments = []
        
        for segment in transcription['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Find the most likely speaker for this segment
            segment_interval = Interval(start_time, end_time)
            speaker_overlaps = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = segment_interval & turn
                if overlap:
                    overlap_duration = overlap.duration
                    if speaker not in speaker_overlaps:
                        speaker_overlaps[speaker] = 0
                    speaker_overlaps[speaker] += overlap_duration
            
            # Assign to speaker with most overlap
            if speaker_overlaps:
                assigned_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            else:
                assigned_speaker = "UNKNOWN"
            
            aligned_segments.append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speaker': assigned_speaker,
                'text': text,
                'confidence': segment.get('avg_logprob', 0)
            })
        
        return aligned_segments
    
    def process_audio_file(self, input_path: str, output_format: str = 'json') -> str:
        """
        Process an M4A audio file and perform complete diarization workflow.
        
        Args:
            input_path: Path to the M4A audio file
            output_format: Output format ('json', 'csv', 'txt')
            
        Returns:
            Path to the output file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Convert M4A to WAV
        wav_path = self.convert_m4a_to_wav(input_path)
        
        try:
            # Perform diarization
            speakers, diarization = self.perform_diarization(wav_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(wav_path)
            
            # Align transcription with speakers
            aligned_segments = self.align_transcription_with_speakers(transcription, diarization)
            
            # Generate output
            output_path = self._save_results(aligned_segments, input_path, output_format)
            
            return output_path
            
        finally:
            # Clean up temporary WAV file
            if os.path.exists(wav_path) and wav_path != input_path:
                os.remove(wav_path)
                print(f"Cleaned up temporary file: {wav_path}")
    
    def _save_results(self, segments: List[Dict], input_path: str, output_format: str) -> str:
        """
        Save results in the specified format.
        
        Args:
            segments: List of aligned segments
            input_path: Original input file path
            output_format: Output format
            
        Returns:
            Path to the output file
        """
        base_name = Path(input_path).stem
        
        if output_format.lower() == 'json':
            output_path = f"{base_name}_diarized.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'input_file': input_path,
                        'total_segments': len(segments),
                        'speakers': list(set(seg['speaker'] for seg in segments))
                    },
                    'segments': segments
                }, f, indent=2, ensure_ascii=False)
        
        elif output_format.lower() == 'csv':
            output_path = f"{base_name}_diarized.csv"
            df = pd.DataFrame(segments)
            df.to_csv(output_path, index=False)
        
        elif output_format.lower() == 'txt':
            output_path = f"{base_name}_diarized.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Audio Diarization Results for: {input_path}\n")
                f.write("=" * 50 + "\n\n")
                
                for segment in segments:
                    start_min = int(segment['start'] // 60)
                    start_sec = int(segment['start'] % 60)
                    end_min = int(segment['end'] // 60)
                    end_sec = int(segment['end'] % 60)
                    
                    f.write(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] ")
                    f.write(f"{segment['speaker']}: {segment['text']}\n")
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"Results saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Audio Diarization with Whisper")
    parser.add_argument("input_file", help="Path to the M4A audio file")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--output-format", default="json",
                       choices=["json", "csv", "txt"],
                       help="Output format (default: json)")
    parser.add_argument("--device", choices=["cuda", "cpu"],
                       help="Device to use (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    try:
        diarizer = AudioDiarizer(whisper_model=args.model, device=args.device)
        output_path = diarizer.process_audio_file(args.input_file, args.output_format)
        print(f"\n✅ Diarization completed successfully!")
        print(f"📄 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 