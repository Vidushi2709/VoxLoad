"""
pipeline_baseline.py
--------------------
Establish speaker baseline from audio recording.

Usage
-----
python pipeline_baseline.py --input baseline.mp4 --speaker spkr_01
python pipeline_baseline.py -i baseline.mp4 -s spkr_01 -m google/gemini-flash-1.5
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from utils import (
    OUTPUT_DIR,
    OUTPUT_DIR_WAV,
    mp4_to_wav,
    transcribe_wav,
    make_openai_client,
    save_baseline,
    run_agents,
)

load_dotenv()


async def baseline_mode(input_path: Path, speaker_id: str, model: str = None) -> dict:
    """
    Establish baseline: transcribe audio, run agents, save baseline.
    
    Parameters
    ----------
    input_path : Path
        Path to baseline audio file
    speaker_id : str
        Speaker identifier
    model : str, optional
        OpenRouter model override
    
    Returns
    -------
    dict with baseline data
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE MODE")
    print(f"{'='*60}")
    print(f"  Speaker: {speaker_id}")
    print(f"  Input:   {input_path.name}")

    suffix = input_path.suffix.lower()
    if suffix in (".mp4", ".m4a", ".mkv", ".mov", ".avi", ".webm"):
        wav_path = OUTPUT_DIR_WAV / (input_path.stem + "_baseline.wav")
        mp4_to_wav(input_path, wav_path)
    elif suffix in (".wav", ".flac", ".ogg", ".mp3"):
        wav_path = input_path
    else:
        print(f"  WARNING: unknown extension '{suffix}', attempting anyway.")
        wav_path = input_path

    # Transcribe
    print(f"\n[Transcribe]")
    transcript = transcribe_wav(wav_path)
    words = transcript.get("words", [])
    if len(words) >= 2:
        duration_sec = words[-1]["end"] - words[0]["start"]
        print(f"  Duration: {duration_sec:.1f}s ({len(words)} words)")
    else:
        print(f"  Duration: unknown ({len(words)} words)")

    # Run agents without baselines (no z-score normalization yet)
    print(f"\n[Agents]")
    client, model_id = make_openai_client(model)
    result = await run_agents(
        transcript["words"],
        transcript["text"],
        client,
        label="baseline",
        model=model_id,
        speaker_id=speaker_id,
        baselines={},  # Empty baselines = raw scores only
        wav_path=str(wav_path),
    )

    agent_scores = result["agent_scores"]
    
    # Save baseline
    print(f"\n[Save Baseline]")
    save_baseline(
        speaker_id=speaker_id,
        agent_scores=agent_scores,
        transcript_text=transcript["text"],
        output_dir=OUTPUT_DIR
    )

    baseline_data = {
        "speaker_id": speaker_id,
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript["text"],
        "agent_baselines": {
            agent: {
                "raw_score": score.get("raw_score", score.get("score")),
                "score": score.get("score"),
                "metadata": {k: v for k, v in score.items() if k not in ["raw_score", "score"]}
            }
            for agent, score in agent_scores.items()
        }
    }

    print(f"\n[OK] Baseline saved for {speaker_id}")
    print(f"  File: output/baselines/{speaker_id}_baseline.json")
    return baseline_data


def main():
    """Parse args and run baseline mode."""
    parser = argparse.ArgumentParser(
        description="Establish speaker baseline from audio recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic baseline creation
  python pipeline_baseline.py -i baseline.mp4 -s spkr_01

  # With custom LLM model
  python pipeline_baseline.py -i baseline.mp4 -s spkr_01 -m google/gemini-flash-1.5
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to baseline MP4 (or WAV/audio) file",
    )
    parser.add_argument(
        "--speaker", "-s",
        type=str,
        required=True,
        metavar="ID",
        help="Speaker identifier (e.g., spkr_01)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        metavar="MODEL",
        help="OpenRouter model ID (e.g., google/gemini-flash-1.5). "
             "Overrides OPENROUTER_MODEL env var.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(baseline_mode(input_path, args.speaker, args.model))


if __name__ == "__main__":
    main()
