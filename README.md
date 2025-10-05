# lsub

> Local AI-powered subtitle generation and translation

[![PyPI Version](https://img.shields.io/pypi/v/lsub)](https://pypi.org/project/lsub/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lsub)](https://pypi.org/project/lsub/)
[![License](https://img.shields.io/pypi/l/lsub)](LICENSE)
[![UV Friendly](https://img.shields.io/badge/uv-friendly-5A2DAA)](https://docs.astral.sh/uv/)
[![CI Publish](https://img.shields.io/github/actions/workflow/status/fcjr/lsub/publish.yml?label=publish)](https://github.com/fcjr/lsub/actions/workflows/publish.yml)

Extract and translate video subtitles using [Whisper](https://github.com/openai/whisper) and [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M).

## Features

- **Automatic speech recognition** using OpenAI Whisper (auto-detects language)
- **Multi-language translation** using Meta's NLLB-200 model
- **Subtitle embedding** into video files (MP4 or MKV)
- **Multiple subtitle tracks** with proper language naming
- **SRT and ASS format support** (ASS for better Unicode/CJK character rendering)

## Install

Recommended (fast, reproducible):

```bash
uv tool install lsub
```

Run without installing:

```bash
uvx lsub video.mp4 -t en es
```

With pip:

```bash
pip install lsub
```

## Usage

```bash
# Basic usage - extract subtitles (auto-detect language)
lsub video.mp4

# Extract and translate to English and Spanish
lsub video.mp4 -t en es

# Specify source language explicitly
lsub video.mp4 -l zh -t en

# Use different Whisper model (default: turbo)
lsub video.mp4 -m large -t en

# Output as MKV (better subtitle support for Unicode/CJK)
lsub video.mp4 -t en zh -f mkv

# Generate SRT files only (no embedding)
lsub video.mp4 -t en es --srt-only

# Custom output path
lsub video.mp4 -t en -o output_with_subs.mp4
```

## Supported Languages

Translation supports 12 common languages:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi

More languages available in [NLLB-200 documentation](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

## Whisper Models

Available models (trade-off between speed and accuracy):
- `tiny` - Fastest, least accurate
- `base` - Fast, decent accuracy
- `small` - Balanced
- `medium` - Good accuracy, slower
- `large` - Best accuracy, slowest
- `turbo` - Fast and accurate (default)

## Output Formats

- **MP4** (default): Compatible but limited Unicode support for subtitles
- **MKV**: Better subtitle support, recommended for Chinese/Japanese/Korean content

MKV uses ASS format with embedded font information for proper CJK character rendering.

## Development

```bash
uv sync
uv run lsub video.mp4 -t en     # run the CLI using local code

# optional: editable install
uv pip install -e .

./scripts/release.sh            # release a new version
```

## Notes

- First run downloads Whisper model (~1.5GB for turbo) and NLLB-200 model (~1.2GB)
- Models are cached in `~/.cache/huggingface/`
- Requires `ffmpeg` installed on your system
- Generated subtitle files (.srt and .ass) are saved alongside the video
- For best CJK (Chinese/Japanese/Korean) subtitle rendering, use MKV format (`-f mkv`)
