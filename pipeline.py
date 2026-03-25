"""
pipeline.py
-----------
End-to-end cognitive-load pipeline.

Usage
-----
# Batch mode (requires baby-data/transcripts/low|medium|high_transcript.json)
python pipeline.py

# Single-file mode (MP4 or WAV — converts, transcribes, then runs all agents)
python pipeline.py --input path/to/recording.mp4
python pipeline.py --input path/to/recording.mp4 --label participant_01
python pipeline.py -i recording.mp4 -l p01 --model google/gemini-flash-1.5

Flags
-----
--input  / -i   Path to an MP4 (or WAV) file to analyse
--label  / -l   Optional display name for this recording (defaults to filename stem)
--model  / -m   OpenRouter model ID (overrides OPENROUTER_MODEL env var)
--output / -o   Path to results JSON file (default: output/pipeline_results.json)
"""

import argparse
import asyncio
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

from Agents.aggregator_agent import aggregator
from Agents.speech_rate_agent import SpeechRateAgent
from Agents.pause_patterns_agent import PausePatternsAgent
from Agents.filler_words_agent import FillerPatternsAgent
from Agents.semantic_density_agent import SemanticDensityAgent
from Agents.syntactic_complexity_agent import SyntacticComplexityAgent

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR_WAV = Path("output/wav")
OUTPUT_DIR_WAV.mkdir(parents=True, exist_ok=True)
# Fallback WPM calibration range for single-file mode (typical human speech)
SINGLE_FILE_WPM_MIN = 80.0
SINGLE_FILE_WPM_MAX = 220.0

BATCH_DATA_DIR = Path("baby-data/transcripts")
BATCH_LABELS   = ["low", "medium", "high"]


# Helpers 

def mp4_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert any audio/video file to 16 kHz mono WAV using ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"  [convert] {input_path.name} → {output_path.name} ...")
    result = subprocess.run(
        [
            ffmpeg_exe, "-i", str(input_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_path), "-y",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-600:]}")
    print(f"  ✓ WAV saved: {output_path}")
    return output_path


def transcribe_wav(wav_path: Path) -> dict:
    """Transcribe a WAV file with faster-whisper; return {text, words}."""
    print(f"  [transcribe] {wav_path.name} ...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=100, speech_pad_ms=100),
        condition_on_previous_text=False,
        temperature=0.0,
    )
    full_text, words_out = [], []
    for seg in segments:
        full_text.append(seg.text.strip())
        for w in seg.words:
            words_out.append({
                "word":  w.word.strip(),
                "start": round(w.start, 3),
                "end":   round(w.end,   3),
            })
    text = " ".join(full_text)
    print(f"  ✓ {len(words_out)} words transcribed")
    return {"text": text, "words": words_out}


def append_to_results(out_path: Path, entry: dict) -> None:
    """
    Append a new run entry to the JSON results file.
    The file is a JSON array — previous results are NEVER overwritten.
    If the file contains an old dict-format result it is migrated automatically.
    """
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                # Migrate old {low, medium, high} dict → list
                print("  [migration] Converting old results format to list …")
                existing = [{"_migrated": True, "timestamp": "unknown", "data": existing}]
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(entry)
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def make_openai_client(model_override: str = None) -> tuple:
    """Return (AsyncOpenAI client, model_id)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    model = (
        model_override
        or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    )
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return client, model


# Agent runner 

async def run_agents(
    words: list,
    text: str,
    client: AsyncOpenAI,
    *,
    label: str       = "",
    model: str       = "meta-llama/llama-3.1-8b-instruct",
    speech_agent: SpeechRateAgent = None,
) -> dict:
    """
    Run all 5 agents in parallel and return aggregated results.
    Pass a pre-fitted `speech_agent` for batch mode; leave None for single-file
    mode (uses fixed WPM calibration range).
    """
    if speech_agent is None:
        speech_agent = SpeechRateAgent()
        speech_agent.min_wpm = SINGLE_FILE_WPM_MIN
        speech_agent.max_wpm = SINGLE_FILE_WPM_MAX
        speech_agent.fitted  = True

    pause_agent    = PausePatternsAgent()
    filler_agent   = FillerPatternsAgent()
    semantic_agent = SemanticDensityAgent(model=model)
    syntax_agent   = SyntacticComplexityAgent()

    tag = f" [{label}]" if label else ""
    print(f"  [agents]{tag} running …")

    try:
        speech_r, pause_r, filler_r, semantic_r, syntax_r = await asyncio.gather(
            asyncio.to_thread(speech_agent.run,      words),
            asyncio.to_thread(pause_agent.compute,   words),
            asyncio.to_thread(filler_agent.compute,  text),
            semantic_agent.compute(text, client),
            asyncio.to_thread(syntax_agent.compute,  text),
        )
    except Exception as e:
        print(f"  ERROR in agents: {e}")
        raise

    agent_scores = {
        "speech_rate":          speech_r,
        "pause_patterns":       pause_r,
        "filler_words":         filler_r,
        "semantic_density":     semantic_r,
        "syntactic_complexity": syntax_r,
    }
    agg = aggregator(agent_scores, words=words, text=text)

    print(f"    speech_rate         : {speech_r['score']}  (wpm={speech_r['wpm']})")
    print(f"    pause_patterns      : {pause_r['score']}   (pauses={pause_r['pause_count']}, mean={pause_r['mean_pause_ms']}ms)")
    print(f"    filler_words        : {filler_r['score']}  (count={filler_r['total_fillers']}, rate={filler_r['filler_rate']:.2%})")
    print(f"    semantic_density    : {semantic_r['score']}  ({semantic_r.get('reasoning', '')[:60]})")
    print(f"    syntactic_complexity: {syntax_r['score']}  "
          f"(sent_len={syntax_r['avg_sentence_len']}, depth={syntax_r['avg_tree_depth']}, "
          f"sub={syntax_r['subordination_rate']:.2%}, method={syntax_r['method']})")
    print(f"    ── LOAD SCORE       : {agg['load_score']}  [{agg['load_label'].upper()}]"
          + (f"  (interaction Δ=+{agg['interaction_delta']})" if agg.get('interaction_delta', 0) > 0 else ""))
    print(f"    ── CONFIDENCE       : {agg['confidence']}  [{agg['confidence_label'].upper()}]  "
          f"(spread={agg['diagnostics']['score_spread']}, "
          f"words={agg['diagnostics']['word_count']}, "
          f"dur={agg['diagnostics']['duration_sec']}s)")
    _pen = agg['penalties']
    print(f"       penalties        : short={_pen['short_audio']}  noise={_pen['noisy_transcript']}  "
          f"agree={_pen['signal_disagreement']}  sem={_pen['semantic_unreliable']}  "
          f"coh={_pen['topic_incoherence']}")

    return {
        "agent_scores":    agent_scores,
        "aggregated_score": agg,
    }


# Pipeline modes 

async def run_single(input_path: Path, label: str, model: str, out_path: Path) -> None:
    """
    Full pipeline for a single MP4 (or WAV) file:
      convert → transcribe → run agents → append result
    """
    print(f"\n{'='*60}")
    print(f"  Single-file mode: {input_path.name}")
    print(f"{'='*60}")

    suffix = input_path.suffix.lower()

    if suffix in (".mp4", ".m4a", ".mkv", ".mov", ".avi", ".webm"):
        wav_path = OUTPUT_DIR_WAV / (input_path.stem + "_converted.wav")
        mp4_to_wav(input_path, wav_path)
    elif suffix in (".wav", ".flac", ".ogg", ".mp3"):
        wav_path = input_path        # already a compatible audio file
    else:
        print(f"  WARNING: unknown extension '{suffix}', attempting transcription anyway.")
        wav_path = input_path

    transcript    = transcribe_wav(wav_path)
    client, model = make_openai_client(model)
    result        = await run_agents(
        transcript["words"], transcript["text"], client,
        label=label, model=model,
    )

    entry = {
        "timestamp":   datetime.now().isoformat(),
        "mode":        "single",
        "label":       label,
        "source_file": str(input_path.resolve()),
        "transcript":  transcript["text"],
        **result,
    }

    append_to_results(out_path, entry)
    print(f"\n✓ Result appended → {out_path}")


async def run_batch(model: str, out_path: Path) -> None:
    """
    Batch mode: process low / medium / high transcripts
    and verify that scores are monotonically ordered.
    """
    print(f"\n{'='*60}")
    print("  Batch mode (low / medium / high)")
    print(f"{'='*60}")

    # Load transcripts
    all_data: dict = {}
    for label in BATCH_LABELS:
        path = BATCH_DATA_DIR / f"{label}_transcript.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {path}\n"
                "Run convert.py + transcribe.py first, or use --input for single-file mode."
            )
        all_data[label] = json.load(open(path))

    # Fit speech-rate agent once on the full dataset
    speech_agent = SpeechRateAgent()
    speech_agent.fit([all_data[l]["words"] for l in BATCH_LABELS])

    client, model = make_openai_client(model)
    batch_results  = {}

    for label in BATCH_LABELS:
        print(f"\nProcessing [{label.upper()}] …")
        result = await run_agents(
            all_data[label]["words"],
            all_data[label]["text"],
            client,
            label=label,
            model=model,
            speech_agent=speech_agent,
        )
        batch_results[label] = {
            **result,
            "ground_truth": BATCH_LABELS.index(label) + 1,
        }
        agg = batch_results[label]["aggregated_score"]["load_score"]
        gt  = batch_results[label]["ground_truth"]
        print(f"    ground_truth   : {gt}/3")
        print(f"    ── LOAD SCORE  : {agg}")

    # Monotonicity check
    scores  = {l: batch_results[l]["aggregated_score"]["load_score"] for l in BATCH_LABELS}
    ordered = scores["low"] < scores["medium"] < scores["high"]
    print(f"\n  Evaluation:")
    print(f"    low={scores['low']}  medium={scores['medium']}  high={scores['high']}")
    print(f"    Correctly ordered low < medium < high: {ordered}")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode":      "batch",
        "results":   batch_results,
        "scores":    scores,
        "ordered":   ordered,
    }
    append_to_results(out_path, entry)
    print(f"\n✓ Results appended → {out_path}")


# Entry point 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cognitive Load Pipeline — analyse an MP4/WAV or run batch mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python pipeline.py                                               # batch mode
        python pipeline.py -i recording.mp4                            # single file
        python pipeline.py -i recording.mp4 -l participant_01          # with label
        python pipeline.py -i recording.mp4 -m google/gemini-flash-1.5 # custom LLM
        python pipeline.py -i recording.mp4 -o my_results.json         # custom output
                """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to an MP4 (or WAV/audio) file to analyse",
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        default=None,
        metavar="NAME",
        help="Display name / label for this recording (defaults to filename stem)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        metavar="MODEL",
        help="OpenRouter model ID (e.g. google/gemini-flash-1.5). "
             "Overrides the OPENROUTER_MODEL env var.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_DIR / "pipeline_results.json"),
        metavar="FILE",
        help="Path to the JSON results file (default: output/pipeline_results.json). "
             "Results are always APPENDED — never overwritten.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        inp = Path(args.input)
        if not inp.exists():
            print(f"ERROR: File not found: {inp}", file=sys.stderr)
            sys.exit(1)
        label = args.label or inp.stem
        asyncio.run(run_single(inp, label=label, model=args.model, out_path=out_path))
    else:
        asyncio.run(run_batch(model=args.model, out_path=out_path))


if __name__ == "__main__":
    main()