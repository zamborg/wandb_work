# Audio Diarization with Whisper

A Python script that performs speaker diarization on M4A audio files using OpenAI's Whisper for transcription and pyannote-audio for speaker identification.

## Features

- **M4A Support**: Handles M4A audio files directly
- **Speaker Diarization**: Identifies different speakers in the audio
- **Whisper Transcription**: High-quality speech-to-text using OpenAI's Whisper
- **Timestamp Alignment**: Provides precise timestamps for each speaker segment
- **Multiple Output Formats**: JSON, CSV, and TXT formats
- **GPU Support**: Automatic GPU detection for faster processing

## Installation

1. **Clone or download the files to your working directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Accept pyannote-audio terms** (required for speaker diarization):
   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the terms and conditions
   - For some models, you may need to set up a HuggingFace token

## Usage

### Basic Usage

```bash
python audio_diarizer.py your_audio_file.m4a
```

### Advanced Options

```bash
# Use a larger Whisper model for better accuracy
python audio_diarizer.py audio.m4a --model large

# Output in CSV format
python audio_diarizer.py audio.m4a --output-format csv

# Force CPU usage
python audio_diarizer.py audio.m4a --device cpu

# Combined options
python audio_diarizer.py audio.m4a --model medium --output-format txt
```

### Command Line Arguments

- `input_file`: Path to the M4A audio file (required)
- `--model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) - default: `base`
- `--output-format`: Output format (`json`, `csv`, `txt`) - default: `json`
- `--device`: Processing device (`cuda`, `cpu`) - auto-detected if not specified

## Output Formats

### JSON Output
```json
{
  "metadata": {
    "input_file": "audio.m4a",
    "total_segments": 15,
    "speakers": ["SPEAKER_00", "SPEAKER_01"]
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "duration": 2.7,
      "speaker": "SPEAKER_00",
      "text": "Hello, how are you today?",
      "confidence": -0.3
    }
  ]
}
```

### CSV Output
Tabular format with columns: start, end, duration, speaker, text, confidence

### TXT Output
Human-readable format:
```
[00:00 - 00:03] SPEAKER_00: Hello, how are you today?
[00:04 - 00:07] SPEAKER_01: I'm doing great, thanks for asking!
```

## Model Information

### Whisper Models
- **tiny**: Fastest, least accurate (~39 MB)
- **base**: Good balance of speed and accuracy (~74 MB)
- **small**: Better accuracy (~244 MB)
- **medium**: High accuracy (~769 MB)
- **large**: Best accuracy (~1550 MB)

### Performance Tips

1. **GPU Usage**: Install PyTorch with CUDA support for faster processing
2. **Model Selection**: Use `base` or `small` for most use cases
3. **Audio Quality**: Higher quality audio produces better diarization results
4. **File Size**: Large files may take several minutes to process

## Troubleshooting

### Common Issues

1. **ImportError**: Install all requirements: `pip install -r requirements.txt`
2. **HuggingFace Token Error**: Accept terms at the pyannote model page
3. **CUDA Issues**: Install PyTorch with appropriate CUDA version
4. **Memory Error**: Try a smaller Whisper model or use CPU

### Audio Format Issues
If you encounter issues with M4A files, try converting to WAV first:
```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- ~2-4GB disk space for models
- ~1-8GB RAM depending on model size and audio length

## License

This project uses several open-source libraries:
- OpenAI Whisper (MIT License)
- pyannote-audio (MIT License)
- PyTorch (BSD License)

## Contributing

Feel free to submit issues and pull requests to improve the script! 