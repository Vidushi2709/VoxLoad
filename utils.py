"""
pipeline_shared.py
------------------
Shared utilities for baseline and test pipelines.

Contains:
  - Audio conversion (mp4_to_wav)
  - Transcription (transcribe_wav)
  - Agent execution (run_agents)
  - Baseline I/O (save_baseline, load_baseline)
  - OpenAI client setup
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

from Agents.pause_patterns_agent import PausePatternsAgent
from Agents.filler_words_agent import FillerPatternsAgent
from Agents.speech_rate_agent import SpeechRateAgent
# from Agents.semantic_density_agent import SemanticDensityAgent  # COMMENTED OUT

load_dotenv()

# Output directories
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR_WAV = Path("output/wav")
OUTPUT_DIR_WAV.mkdir(parents=True, exist_ok=True)


def mp4_to_wav(input_path: Path, output_path: Path) -> Path:
    """Convert any audio/video file to 16 kHz mono WAV using ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"  [convert] {input_path.name} -> {output_path.name} ...")
    _t0 = time.perf_counter()
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
    print(f"  [OK] WAV saved: {output_path}  ({time.perf_counter()-_t0:.1f}s)")
    return output_path


def transcribe_wav(wav_path: Path) -> dict:
    """Transcribe a WAV file with faster-whisper; return {text, words}."""
    print(f"  [transcribe] {wav_path.name} ...")
    _t0 = time.perf_counter()
    print(f"    loading model ...")
    _tm = time.perf_counter()
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print(f"    model loaded  ({time.perf_counter()-_tm:.1f}s)")
    _ti = time.perf_counter()
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
    print(f"  [OK] {len(words_out)} words transcribed  "
          f"(inference: {time.perf_counter()-_ti:.1f}s | total: {time.perf_counter()-_t0:.1f}s)")
    return {"text": text, "words": words_out}


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


def save_baseline(speaker_id: str, agent_scores: dict, transcript_text: str = "", output_dir: Path = OUTPUT_DIR) -> Path:
    """
    Save raw agent scores as baseline for a speaker.
    
    Parameters
    ----------
    speaker_id : str
        Speaker identifier (e.g., spkr_01)
    agent_scores : dict
        {agent_name: {score, raw_score, ...}, ...}
    transcript_text : str, optional
        Baseline transcript for future comparison
    output_dir : Path
        Directory to save baselines
    
    Returns
    -------
    Path to saved baseline file
    """
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_file = baseline_dir / f"{speaker_id}_baseline.json"
    
    # Extract raw_score from each agent
    baseline_data = {
        "speaker_id": speaker_id,
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript_text,
        "agent_baselines": {
            agent: {
                "raw_score": scores.get("raw_score", scores.get("score")),
                "score": scores.get("score"),  # z-score (should be ~0 for baseline)
            }
            for agent, scores in agent_scores.items()
        }
    }
    
    baseline_file.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    print(f"  [OK] Baseline saved -> {baseline_file}")
    return baseline_file


def load_baseline(speaker_id: str, output_dir: Path = OUTPUT_DIR) -> dict:
    """
    Load baseline for a speaker.
    
    Parameters
    ----------
    speaker_id : str
        Speaker identifier
    output_dir : Path
        Directory where baselines are stored
    
    Returns
    -------
    dict with agent baselines and transcript, or empty dict if not found
    """
    baseline_file = output_dir / "baselines" / f"{speaker_id}_baseline.json"
    
    if not baseline_file.exists():
        raise FileNotFoundError(f"No baseline found for {speaker_id} at {baseline_file}")
    
    data = json.loads(baseline_file.read_text(encoding="utf-8"))
    print(f"  [OK] Baseline loaded from {baseline_file}")
    return data


async def run_agents(
    words: list,
    text: str,
    client: AsyncOpenAI,
    *,
    label: str       = "",
    model: str       = "meta-llama/llama-3.1-8b-instruct",
    speaker_id: str  = None,
    baselines: dict  = None,
    wav_path: str    = None,
    baseline_transcript: str = None,
) -> dict:
    """
    Run all agents in parallel and return raw scores.
    
    Parameters
    ----------
    words, text : transcript data
    client : AsyncOpenAI client
    label : optional display name
    model : LLM model
    speaker_id : speaker identifier for baseline comparison
    baselines : pre-loaded baselines dict
    wav_path : path to audio file (for speech_rate agent)
    baseline_transcript : baseline transcript for semantic_density comparison
    
    Returns
    -------
    {agent_scores, speaker_id, transcript_data}
    """
    pause_agent    = PausePatternsAgent()
    filler_agent   = FillerPatternsAgent()
    speech_agent   = SpeechRateAgent()
    # semantic_agent = SemanticDensityAgent(model=model)  # COMMENTED OUT

    tag = f" [{label}]" if label else ""
    print(f"  [agents]{tag} running …")
    _t_agents = time.perf_counter()

    # time each agent individually via wrappers 
    def _timed(name, fn, *args):
        t0 = time.perf_counter()
        r  = fn(*args)
        print(f"    {name:<22} {time.perf_counter()-t0:.1f}s")
        return r

    async def _timed_async(name, coro):
        t0 = time.perf_counter()
        r  = await coro
        print(f"    {name:<22} {time.perf_counter()-t0:.1f}s")
        return r

    print(f"    {'agent':<22} elapsed")
    print(f"    {'-'*30}")
    try:
        pause_r, filler_r, speech_r = await asyncio.gather(
            asyncio.to_thread(_timed, "pause_patterns",    pause_agent.compute,  words, speaker_id, baselines),
            asyncio.to_thread(_timed, "filler_words",      filler_agent.compute, text, True, speaker_id, baselines),
            asyncio.to_thread(_timed, "speech_rate",       speech_agent.run,     words, wav_path, speaker_id, baselines),
            # _timed_async("semantic_density", semantic_agent.compute(text, client, None, speaker_id, baselines, baseline_transcript)),  # COMMENTED OUT
        )
    except Exception as e:
        print(f"  ERROR in agents: {e}")
        raise

    print(f"    {'-'*30}")
    print(f"    {'agents total':<22} {time.perf_counter()-_t_agents:.1f}s")

    agent_scores = {
        "pause_patterns":   pause_r,
        "filler_words":     filler_r,
        "speech_rate":      speech_r,
        # "semantic_density": semantic_r,  # COMMENTED OUT
    }

    print(f"")
    print(f"    pause_patterns   : raw={pause_r.get('raw_score', pause_r['score']):.3f}  z={pause_r['score']:.3f}  (pauses={pause_r['pause_count']})")
    print(f"    filler_words     : raw={filler_r.get('raw_score', filler_r['score']):.3f}  z={filler_r['score']:.3f}  (count={filler_r['total_fillers']})")
    print(f"    speech_rate      : raw={speech_r.get('raw_score', speech_r['score']):.3f}  z={speech_r['score']:.3f}  (wpm={speech_r.get('wpm', 0):.0f})")
    # print(f"    semantic_density : raw={semantic_r.get('raw_score', semantic_r['score']):.3f}  z={semantic_r['score']:.3f}  ({semantic_r.get('reasoning', '')[:40]})")  # COMMENTED OUT

    return {
        "agent_scores":    agent_scores,
        "speaker_id":      speaker_id or "unknown",
    }
