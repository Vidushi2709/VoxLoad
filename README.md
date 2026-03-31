# VoxLoad: Voice as a Proxy for Cognitive Load

A multi-agent pipeline that estimates **cognitive load** from speech recordings using within-speaker z-score normalization. Give it an MP4 (or WAV) and a speaker baseline, and it returns normalized deviation scores for pause patterns, filler words, and speech rate.

---

## How it works

```
MP4 / WAV
   │
   ▼
[ffmpeg]  →  16 kHz mono WAV
   │
   ▼
[Whisper]  →  word-level transcript + timestamps
   │
   ├──▶  SpeechRateAgent      (WPM + variance)
   ├──▶  PausePatternsAgent   (pause count, duration, long-pause fraction)
   ├──▶  FillerWordsAgent     (um/uh/like/… weighted counts)
   └──▶  SemanticDensityAgent (LLM density — NOT IN USE: unstable)
              │
              ▼
         AggregatorAgent  →  load_score (z-score)  +  confidence (0–1)
```

---

## Key Findings

**1. Per-Speaker Normalization is Non-Negotiable**  
Raw acoustic scores vary wildly across speakers (baseline filler rates ranged 0.195–0.888). Z-score normalization against each speaker's own spontaneous speech baseline is essential for meaningful comparison.

**2. Filler Words Dominates for Low-Baseline Speakers**  
Speakers with naturally low disfluency (ayu, vin) showed clear filler word increase across difficulty levels. For high-baseline speakers (lak), filler words hit a ceiling effect and became an unreliable signal.

**3. Pause Patterns Dominates for High-Baseline Speakers**  
Lak (baseline filler rate 0.888) showed consistent pause pattern escalation across difficulty levels, while filler words remained inverted. Signal dominance is speaker-dependent.

**4. Designed Difficulty ≠ Experienced Difficulty**  
Same question produced opposite acoustic profiles depending on domain familiarity. Ayu's high-difficulty technical question showed near-baseline disfluency; vin's showed maximum load. Acoustic deviation outperformed label-based difficulty at capturing actual cognitive experience.

**5. Speech Rate is Unreliable**  
No consistent directional pattern across speakers or difficulty levels. More sensitive to individual speaking style than cognitive load.

---

## Agents

| Agent | Method | What it measures | Status |
|-------|--------|-----------------|--------|
| **SpeechRateAgent** | Algorithmic | Words-per-minute and its sliding-window variance. Slower + more variable = higher load. | ✓ Active |
| **PausePatternsAgent** | Algorithmic | Pause rate (per min), mean pause duration, and fraction of long pauses (>1 s). | ✓ Active |
| **FillerWordsAgent** | Regex / spaCy | `um`, `uh`, `er` etc. counted at 2× weight; hedges (`like`, `actually`, `well`, …) at 1×. | ✓ Active |
| **SemanticDensityAgent** | LLM (OpenRouter) | Asks an LLM to rate how information-dense the speech is; inverted so vague = high load. | ✗ Not in use (unstable) |

Aggregator weights (active agents): `pause_patterns 41% · speech_rate 35% · filler_words 29%`

---

## Setup

```bash
# 1. Clone & enter
git clone https://github.com/Vidushi2709/VoxLoad
cd VoxLoad

# 2. Install dependencies (uv recommended)
uv sync
# or: pip install -r requirements.txt

# 3. Set your LLM API key
echo "API_KEY=sk-or-..." > .env
```

---

## Usage

### Analyse a single file with speaker baseline
```bash
python pipeline.py -i path/to/recording.mp4 -s speaker_name -b baseline.json
```

### Batch analysis with speaker metadata
```bash
python pipeline.py -i recordings/ -m metadata.csv
```
Generate speaker baselines first via `baseline.py` on spontaneous speech recordings.

### Generate z-score report
```bash
python z_score.py -r output/test_results/ -b output/baselines/
```
Compares all results against per-speaker baseline, outputs deviation report.

---

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--input` | `-i` | Path to audio file or directory (MP4, WAV, M4A, MKV, MOV, FLAC, MP3) |
| `--speaker` | `-s` | Speaker ID for baseline lookup and per-speaker normalization |
| `--baseline` | `-b` | Path to speaker baseline JSON file (required for normalization) |
| `--metadata` | `-m` | CSV with speaker/difficulty/domain labels for batch analysis |
| `--output` | `-o` | Output directory for deviation scores and reports |

---

## Project Structure

```
speech-cognitive-load/
├── pipeline.py              # Main analysis entry point
├── baseline.py              # Generate per-speaker baselines
├── z_score.py               # Compute z-score deviations & generate reports
├── utils.py                 # Shared utilities (transcription, audio processing)
├── Agents/
│   ├── __init__.py
│   ├── pause_patterns_agent.py
│   ├── filler_words_agent.py
│   ├── speech_rate_agent.py
│   ├── aggregator_agent.py
│   └── confidence_score.py
├── Data/
│   ├── label.csv            # Question metadata (difficulty, domain)
│   ├── questions.csv        # Question bank
│   ├── baseline/            # Speaker spontaneous speech recordings
│   ├── speaker_01/          # Test responses
│   ├── speaker_02/
│   └── speaker_03/
├── output/
│   ├── baselines/           # Speaker baseline JSON files (outputs)
│   └── test_results/        # Deviation scores by speaker (outputs)
└── .env                     # API keys (not committed)
```

---

## Data Format

Results are stored as JSON files with per-speaker deviation scores:

```json
{
  "speaker": "speaker_id",
  "baseline_filler_rate": 0.195,
  "responses": [
    {
      "question_id": "q01",
      "difficulty": "low",
      "domain": "Personal",
      "pause_z": 0.013,
      "filler_z": -0.003,
      "rate_z": -0.015
    }
  ]
}
```

---

## Conclusion

Voice is a **partially reliable proxy for cognitive load** when properly calibrated. Key takeaways:

1. **Per-speaker baseline normalization is non-negotiable** — raw acoustic scores vary 4–5× across speakers. Without baseline calibration, no meaningful comparison is possible.

2. **Signal dominance is speaker-dependent** — filler words reliably detects load increase for low-baseline speakers; pause patterns reliably detects it for high-baseline speakers. A one-size-fits-all weighting will fail silently for half your speakers.

3. **Acoustic deviation beats label-based difficulty** — the same designed question produced opposite acoustic profiles depending on speaker domain familiarity. Deviation scores captured actual experienced load; labels did not.

4. **Multi-signal approach is essential** — no single feature (pause, filler, rate, density) is universally predictive. Combining pause patterns + filler words + speaker-relative normalization produced the most reliable load estimates in this 17-response preliminary study.

For production use: **collect speaker baselines, normalize within-speaker, weight signals adaptively, trust acoustic deviation over label-based difficulty assessments.**

See [output.md](output.md) for detailed findings, results tables, and methodology.
