"""
pipeline_test.py
----------------
Test audio against established speaker baseline.

Usage
-----
python pipeline_test.py --input test.mp4 --speaker spkr_01
python pipeline_test.py -i test.mp4 -s spkr_01 -m google/gemini-flash-1.5
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
    load_baseline,
    run_agents,
)

load_dotenv()


async def test_mode(input_path: Path, speaker_id: str, model: str = None) -> dict:
    """
    Test mode: load baseline, transcribe test audio, run agents,
    compute deviations, return comparison.
    
    Parameters
    ----------
    input_path : Path
        Path to test audio file
    speaker_id : str
        Speaker identifier (must have baseline)
    model : str, optional
        OpenRouter model override
    
    Returns
    -------
    dict with baseline comparison and deviations
    """
    print(f"\n{'='*60}")
    print(f"  TEST MODE")
    print(f"{'='*60}")
    print(f"  Speaker: {speaker_id}")
    print(f"  Input:   {input_path.name}")

    # Load baseline
    print(f"\n[Load Baseline]")
    try:
        baseline = load_baseline(speaker_id, OUTPUT_DIR)
        print(f"  Baseline loaded: {baseline['timestamp']}")
    except FileNotFoundError:
        print(f"  ERROR: No baseline found for {speaker_id}")
        print(f"  File: output/baselines/{speaker_id}_baseline.json")
        print(f"  Run baseline mode first: python pipeline_baseline.py -i FILE -s {speaker_id}")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix in (".mp4", ".m4a", ".mkv", ".mov", ".avi", ".webm"):
        wav_path = OUTPUT_DIR_WAV / (input_path.stem + "_test.wav")
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

    # Run agents WITH baselines (z-score computation enabled)
    print(f"\n[Agents]")
    client, model_id = make_openai_client(model)
    
    # Restructure baseline data: extract raw_score per agent
    agent_baselines_dict = baseline.get("agent_baselines", {})
    baselines_formatted = {
        speaker_id: {
            agent: agent_baselines_dict[agent].get("raw_score", 0.5)
            for agent in agent_baselines_dict
        }
    }
    baseline_transcript = baseline.get("transcript", "")
    
    result = await run_agents(
        transcript["words"],
        transcript["text"],
        client,
        label="test",
        model=model_id,
        speaker_id=speaker_id,
        baselines=baselines_formatted,  # Now: {speaker_id: {agent: value, ...}}
        wav_path=str(wav_path),
        baseline_transcript=baseline_transcript,
    )

    agent_scores = result["agent_scores"]

    # Compare baseline vs test
    print(f"\n[Comparison]")
    comparison = {
        "speaker_id": speaker_id,
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "test": {
            "transcript": transcript["text"],
            "agent_scores": {
                agent: {
                    "raw_score": score.get("raw_score", score.get("score")),
                    "z_score": score.get("score"),
                }
                for agent, score in agent_scores.items()
            }
        },
        "deviations": {}
    }

    # Compute deviations
    for agent in agent_scores.keys():
        baseline_raw = baseline["agent_baselines"][agent]["raw_score"]
        test_raw = agent_scores[agent].get("raw_score", agent_scores[agent].get("score"))
        
        comparison["deviations"][agent] = {
            "raw_delta": test_raw - baseline_raw,
            "z_score": agent_scores[agent].get("score"),
        }
        
        print(f"  {agent:25} baseline={baseline_raw:.3f}  test={test_raw:.3f}  delta={test_raw - baseline_raw:+.3f}  z={agent_scores[agent].get('score'):.3f}")

    return comparison


def main():
    """Parse args and run test mode."""
    parser = argparse.ArgumentParser(
        description="Test audio against speaker baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test against baseline
  python pipeline_test.py -i test.mp4 -s spkr_01

  # With custom LLM model
  python pipeline_test.py -i test.mp4 -s spkr_01 -m google/gemini-flash-1.5

Note:
  Must have created a baseline first using pipeline_baseline.py
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to test MP4 (or WAV/audio) file",
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

    asyncio.run(test_mode(input_path, args.speaker, args.model))


if __name__ == "__main__":
    main()
