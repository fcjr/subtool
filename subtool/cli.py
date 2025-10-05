import argparse
import os
import sys
import tempfile
from pathlib import Path
import warnings

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ffmpeg

# Suppress warnings
warnings.filterwarnings('ignore')

# Language code mapping for NLLB-200
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'zh': 'zho_Hans',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
}

# Language names for metadata
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
}

def extract_audio(video_path, audio_path):
    """Extract audio from video file using ffmpeg"""
    try:
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',
            ac=1,
            ar='16k'
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        sys.exit(1)

def transcribe_audio(audio_path, model_name='turbo', language=None):
    """Transcribe audio using Whisper"""
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    # Validate language code if provided
    if language:
        valid_languages = whisper.tokenizer.LANGUAGES.keys()
        if language not in valid_languages:
            print(f"Error: Invalid language code '{language}'")
            print(f"Valid codes: {', '.join(sorted(valid_languages))}")
            sys.exit(1)

    print("Transcribing audio...")
    if language:
        result = model.transcribe(audio_path, language=language)
        print(f"Transcribing as: {language}")
    else:
        result = model.transcribe(audio_path)
        print(f"Detected language: {result['language']}")

    return result

def write_srt(segments, output_path):
    """Write segments to SRT subtitle file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

def write_ass(segments, output_path, font_name="Arial Unicode MS"):
    """Write segments to ASS subtitle file with font support"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write ASS header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n\n")

        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{font_name},48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2.5,1.5,2,10,10,35,1\n\n")

        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for segment in segments:
            start = format_timestamp_ass(segment['start'])
            end = format_timestamp_ass(segment['end'])
            text = segment['text'].strip().replace('\n', '\\N')

            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

def format_timestamp_ass(seconds):
    """Convert seconds to ASS timestamp format (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def translate_segments(segments, src_lang, tgt_lang, model, tokenizer):
    """Translate subtitle segments using NLLB-200"""
    translated_segments = []

    # Set source language
    tokenizer.src_lang = src_lang

    for segment in segments:
        text = segment['text'].strip()

        # Translate text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512
        )
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # Create new segment with translated text
        translated_segment = {
            'start': segment['start'],
            'end': segment['end'],
            'text': translation
        }
        translated_segments.append(translated_segment)

    return translated_segments

def find_system_font():
    """Find a suitable CJK font on the system"""
    import platform
    import os

    font_paths = []

    if platform.system() == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    elif platform.system() == 'Linux':
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        ]
    elif platform.system() == 'Windows':
        font_paths = [
            'C:\\Windows\\Fonts\\msyh.ttc',  # Microsoft YaHei
            'C:\\Windows\\Fonts\\arial.ttf',
        ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path

    return None

def embed_subtitles(video_path, subtitle_files, output_path):
    """Embed subtitle files into video as optional streams using ffmpeg with font attachment"""
    print(f"Embedding {len(subtitle_files)} subtitle track(s)...")

    try:
        # Detect output format
        is_mp4 = str(output_path).lower().endswith('.mp4')

        # Find a system font for embedding (MKV only)
        font_path = find_system_font() if not is_mp4 else None

        # Build ffmpeg command with multiple inputs
        input_args = ['-i', video_path]

        # Add each subtitle file as an input
        for sub_file, lang in subtitle_files:
            input_args.extend(['-i', sub_file])

        # Add font file if found and using MKV
        if font_path and not is_mp4:
            input_args.extend(['-attach', font_path, '-metadata:s:t:0', 'mimetype=font/ttf'])

        # Map video and audio from original file
        map_args = ['-map', '0:v', '-map', '0:a?']

        # Map each subtitle stream
        for i in range(len(subtitle_files)):
            map_args.extend(['-map', f'{i+1}:0'])

        # Set subtitle codec based on container format
        if is_mp4:
            # MP4 requires mov_text codec
            codec_args = ['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text']
        else:
            # MKV can use SRT natively
            codec_args = ['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'copy']

        # Add language metadata and title for each subtitle track
        metadata_args = []
        for i, (sub_file, lang) in enumerate(subtitle_files):
            lang_name = LANGUAGE_NAMES.get(lang, lang.upper())
            metadata_args.extend([f'-metadata:s:s:{i}', f'language={lang}'])

            # Set both title and handler_name for maximum compatibility
            metadata_args.extend([f'-metadata:s:s:{i}', f'title={lang_name}'])
            if is_mp4:
                metadata_args.extend([f'-metadata:s:s:{i}', f'handler_name={lang_name}'])

        # Combine all arguments
        cmd_args = ['ffmpeg'] + input_args + map_args + codec_args + metadata_args + ['-y', output_path]

        # Run ffmpeg command
        import subprocess
        result = subprocess.run(cmd_args, capture_output=True)

        if result.returncode != 0:
            print(f"Error embedding subtitles: {result.stderr.decode()}")
            sys.exit(1)

    except Exception as e:
        print(f"Error embedding subtitles: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Extract and translate video subtitles using Whisper and NLLB-200',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Available language codes:
  {', '.join(f'{k}={LANGUAGE_NAMES[k]}' for k in sorted(LANGUAGE_CODES.keys()))}

Whisper models (speed vs accuracy):
  tiny   - Fastest, least accurate
  base   - Fast, decent accuracy
  small  - Balanced (good for most cases)
  medium - High accuracy, slower
  large  - Best accuracy, slowest
  turbo  - Fast and accurate (default)

Output formats:
  mp4 - Default, widely compatible (limited Unicode subtitle support)
  mkv - Better subtitle support, recommended for Chinese/Japanese/Korean

Examples:
  # Extract subtitles (auto-detect language)
  subtitle video.mp4

  # Extract and translate to English and Spanish
  subtitle video.mp4 -t en es

  # Specify source language and Whisper model
  subtitle video.mp4 -l zh -m large -t en

  # Use MKV for better CJK character support
  subtitle video.mp4 -t zh ja -f mkv

  # Generate SRT files only (no embedding)
  subtitle video.mp4 -t en es --srt-only

  # Custom output path
  subtitle video.mp4 -t en -o output.mkv
        '''
    )

    parser.add_argument('video', help='Input video file')
    parser.add_argument('-o', '--output',
                       help='Output video file (default: input_with_subs.mp4)')
    parser.add_argument('-m', '--model', default='turbo',
                       help='Whisper model: tiny, base, small, medium, large, turbo (default: turbo)')
    parser.add_argument('-l', '--lang', '--language', dest='language',
                       help='Source language code for Whisper (auto-detected if not specified)')
    parser.add_argument('-t', '--translate', nargs='+', metavar='LANG',
                       help='Translate to language codes (e.g., en es fr). Source language is always included.')
    parser.add_argument('-f', '--format', choices=['mkv', 'mp4'], default='mp4',
                       help='Output format (default: mp4). Use mkv for better CJK subtitle support.')
    parser.add_argument('--srt-only', action='store_true',
                       help='Generate SRT/ASS files only without embedding in video')

    args = parser.parse_args()

    # Validate Whisper model
    valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
    if args.model not in valid_models:
        print(f"Error: Invalid Whisper model '{args.model}'")
        print(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)

    # Validate input file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)

    # Set output path and infer format from extension if output is specified
    format_explicitly_set = '-f' in sys.argv or '--format' in sys.argv

    if args.output:
        output_path = Path(args.output)
        # Infer format from output extension only if -f/--format was not explicitly set
        if not format_explicitly_set:
            output_ext = output_path.suffix.lower().lstrip('.')
            if output_ext in ['mkv', 'mp4']:
                args.format = output_ext
    else:
        output_path = video_path.parent / f"{video_path.stem}_with_subs.{args.format}"

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        audio_path = temp_dir / "audio.wav"

        # Step 1: Extract audio from video
        print("Extracting audio from video...")
        extract_audio(str(video_path), str(audio_path))

        # Step 2: Transcribe audio with Whisper
        result = transcribe_audio(str(audio_path), args.model, args.language)

        # Get source language
        src_lang = result['language']
        if src_lang not in LANGUAGE_CODES:
            src_lang_code = 'eng_Latn'  # Default fallback
        else:
            src_lang_code = LANGUAGE_CODES[src_lang]

        # Step 3: Write original subtitles (SRT + ASS for embedding)
        original_srt = video_path.parent / f"{video_path.stem}.{src_lang}.srt"
        original_ass = video_path.parent / f"{video_path.stem}.{src_lang}.ass"
        print(f"Writing subtitles to {original_srt.name}...")
        write_srt(result['segments'], original_srt)
        write_ass(result['segments'], original_ass)

        # Use ASS for MKV (better font support), SRT for MP4
        subtitle_file = original_ass if args.format == 'mkv' else original_srt
        subtitle_files = [(str(subtitle_file), src_lang)]

        # Step 4: Translate if requested
        if args.translate:
            # Filter out source language from translation targets
            languages_to_translate = [lang for lang in args.translate if lang != src_lang]

            if languages_to_translate:
                print("Loading translation model...")
                model_name = "facebook/nllb-200-distilled-600M"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                for target_lang in languages_to_translate:
                    if target_lang not in LANGUAGE_CODES:
                        print(f"Warning: Unknown language code '{target_lang}', skipping...")
                        continue

                    tgt_lang_code = LANGUAGE_CODES[target_lang]
                    print(f"Translating to {target_lang}...")

                    translated_segments = translate_segments(
                        result['segments'],
                        src_lang_code,
                        tgt_lang_code,
                        model,
                        tokenizer
                    )

                    # Write translated subtitles (SRT + ASS for embedding)
                    translated_srt = video_path.parent / f"{video_path.stem}.{target_lang}.srt"
                    translated_ass = video_path.parent / f"{video_path.stem}.{target_lang}.ass"
                    write_srt(translated_segments, translated_srt)
                    write_ass(translated_segments, translated_ass)
                    print(f"Wrote {translated_srt.name}")

                    # Use ASS for MKV, SRT for MP4
                    subtitle_file = translated_ass if args.format == 'mkv' else translated_srt
                    subtitle_files.append((str(subtitle_file), target_lang))

            # Inform user if source language was in translation list
            if src_lang in args.translate and src_lang not in languages_to_translate:
                print(f"Skipping translation to {src_lang} (already the source language)")

        # Step 5: Embed subtitles in video (unless --srt-only)
        if not args.srt_only:
            embed_subtitles(str(video_path), subtitle_files, str(output_path))
            print(f"✓ Video with subtitles saved to {output_path}")
        else:
            print(f"✓ Generated {len(subtitle_files)} subtitle file(s)")

if __name__ == "__main__":
    main()
