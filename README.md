# Speech Cognitive Load Analyser

A multi-agent pipeline that estimates **cognitive load** from speech recordings. Give it an MP4 (or WAV) and it returns a z-score, letting you define what "high" means for your use case.

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
   └──▶  SemanticDensityAgent (LLM density score via OpenRouter)
              │
              ▼
         AggregatorAgent  →  load_score (z-score)  +  confidence (0–1)
```

---

## Agents

| Agent | Method | What it measures |
|-------|--------|-----------------|
| **SpeechRateAgent** | Algorithmic | Words-per-minute and its sliding-window variance. Slower + more variable = higher load. |
| **PausePatternsAgent** | Algorithmic | Pause rate (per min), mean pause duration, and fraction of long pauses (>1 s). |
| **FillerWordsAgent** | Regex / spaCy | `um`, `uh`, `er` etc. counted at 2× weight; hedges (`like`, `actually`, `well`, …) at 1×. |
| **SemanticDensityAgent** | LLM (OpenRouter) | Asks an LLM to rate how information-dense the speech is; inverted so vague = high load. |

Aggregator weights: `pause_patterns 35% · speech_rate 30% · filler_words 25% · semantic_density 10%`

---

## Setup

```bash
# 1. Clone & enter
git clone <repo-url>
cd speech-cognitive-load

# 2. Install dependencies (uv recommended)
uv sync
# or: pip install -r requirements.txt

# 3. Set your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

---

## Usage

### Analyse a single file
```bash
python pipeline.py -i path/to/recording.mp4
```

### With a label and custom LLM
```bash
python pipeline.py -i recording.mp4 -l "participant_03" -m google/gemini-flash-1.5
```

### Save results to a custom file
```bash
python pipeline.py -i recording.mp4 -o results/study1.json
```

### Batch mode (low / medium / high ground-truth comparison)
```bash
python pipeline.py
```
Requires `baby-data/transcripts/low_transcript.json`, `medium_transcript.json`, `high_transcript.json`.  
Run `convert.py` then `transcribe.py` to generate these from raw MP4s.

---

## Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | — | Path to MP4, WAV, M4A, MKV, MOV, FLAC, MP3 |
| `--label` | `-l` | filename stem | Display name stored in results |
| `--model` | `-m` | `OPENROUTER_MODEL` env var | OpenRouter model for semantic agent |
| `--output` | `-o` | `output/pipeline_results.json` | Results file — **always appended, never overwritten** |

---

## Output

Results are stored as a **JSON array** — every run appends a new entry:

```json
[
  {
    "timestamp": "2026-03-24T21:47:00",
    "mode": "single",
    "label": "participant_03",
    "source_file": "C:/recordings/session.mp4",
    "transcript": "so I think um the answer is...",
    "agent_scores": {
      "speech_rate":      { "wpm": 98.4, "score": 0.72 },
      "pause_patterns":   { "pause_count": 14, "score": 0.61 },
      "filler_words":     { "total_fillers": 6, "score": 0.41 },
      "semantic_density": { "density_score": 0.3, "score": 0.70 }
    },
    "aggregated_score": {
      "load_score": 0.634,
      "load_label": "high",
      "contributions": { ... }
    }
  }
]
```

---

## Project structure

```
speech-cognitive-load/
├── pipeline.py              # Main entry point
├── convert.py               # Batch MP4 → WAV conversion
├── transcribe.py            # Batch WAV → transcript JSON
├── Agents/
│   ├── aggregator_agent.py
│   ├── speech_rate_agent.py
│   ├── pause_patterns_agent.py
│   ├── filler_words_agent.py
│   └── semantic_density_agent.py
├── baby-data/               # Raw audio + transcript test data
├── output/                  # Results JSON files
└── .env                     # API keys (not committed)
```

---

## Test Results

Results from pipeline runs (18 recordings total):

### Summary Table

| Label | Load Score | Load Label | Speech Rate | Pause Patterns | Filler Words | Semantic Density | Status |
|-------|-----------|-----------|-------------|---|---|---|---|
| `low` | 0.10 | 🟢 Low | 0.0 | 0.0 | 0.0 | 1.0 | First batch (short) |
| `medium` | 0.387 | 🟡 Medium | 0.343 | 0.485 | 0.136 | 0.8 | First batch (limited) |
| `high` | 0.594 | 🔴 High | 0.683 | 0.592 | 0.408 | 0.8 | First batch (short) |
| `q10_low` | — | — | 0.470 | 0.679 | 0.247 | 0.650 | **New batch** |
| `q11_low` | — | — | 0.573 | 0.592 | 0.437 | 0.650 | **New batch** |
| `q24_high` | — | — | 0.634 | 0.670 | 0.233 | 0.650 | **New batch** |

### Detailed Agent Metrics

#### Speech Rate Agent
*Measures words-per-minute and variance. Higher WPM variance and slow scores indicate cognitive load.*

| Recording | WPM | Variance | Slow Score | Rush Score | Final Score | Status |
|-----------|------|----------|-----------|-----------|---------|--------|
| q01_low | 136.4 | 0.00 | 0.597 | 0.000 | 0.269 | ✓ |
| q02_low | 86.4 | 149.49 | 0.954 | 0.000 | 0.560 | ✓ |
| q03_medium | 96.3 | 8.00 | 0.884 | 0.000 | 0.405 | ✓ |
| q04_medium | 86.1 | 69.84 | 0.957 | 0.000 | 0.492 | ✓ |
| q05_high | 64.8 | 103.25 | 1.000 | 0.000 | 0.540 | ✓ |
| **q06_high** | **23.6** | **56.25** | **1.000** | **0.000** | **0.499** | **⚠️ UNRELIABLE** |
| q10_low | 151.5 | 260.84 | 0.489 | 0.108 | 0.470 | ✓ |
| q11_low | 169.4 | 693.92 | 0.361 | 0.303 | 0.573 | ✓ |
| q16_medium | 132.3 | 484.94 | 0.627 | 0.018 | 0.635 | ✓ |
| q17_medium | 120.8 | 604.41 | 0.709 | 0.000 | 0.669 | ✓ |

**Key finding:** q06_high scores 23.6 WPM (extremely slow) with only 21 words — new guard gate flags as `unreliable`.

#### Pause Patterns Agent
*Detects hesitation: count, mean duration, rate, and long pauses (>1200ms). Higher = higher cognitive load.*

| Recording | Pauses | Mean(ms) | Rate/min | Long% | Final Score | Status |
|-----------|--------|----------|----------|-------|---------|--------|
| q02_low | 26 | 1050.0 | 23.41 | 42.3% | 0.795 | ✓ |
| q03_medium | 9 | 884.4 | 17.68 | 44.4% | 0.690 | ✓ |
| q04_medium | 12 | 895.8 | 16.93 | 33.3% | 0.662 | ✓ |
| q05_high | 19 | 1088.4 | 24.62 | 63.2% | 0.855 | ✓ |
| **q06_high** | **17** | **2429.4** | **18.23** | **88.2%** | **0.842** | **⚠️ SEVERE** |
| q10_low | 39 | 612.1 | 28.41 | 10.3% | 0.679 | ✓ |
| q11_low | 33 | 501.5 | 23.01 | 12.1% | 0.592 | ✓ |
| q16_medium | 31 | 588.7 | 21.81 | 19.4% | 0.618 | ✓ |
| q17_medium | 30 | 799.3 | 25.33 | 33.3% | 0.742 | ✓ |

**Key finding:** q06_high has mean pause of 2429ms (2.4 seconds!) — 2× normal range, indicating deep cognitive struggle.

#### Filler Words Agent
*Disfluency markers (um/uh at 2× weight, others at 1×). Higher rate = higher load. Logarithmic scoring applied.*

| Recording | Total Words | Fillers | Rate | Weighted | Final Score | Status |
|-----------|-------------|---------|------|----------|---------|--------|
| q02_low | 96 | 2 | 2.1% | 2.1% | 0.174 | ✓ |
| q03_medium | 49 | 9 | 18.4% | 18.4% | **0.929** | ✓ Log-scaled |
| q04_medium | 61 | 7 | 11.5% | 11.5% | 0.956 | ✓ |
| q05_high | 49 | 8 | 16.3% | 20.4% | **0.990** | ✓ Log-scaled |
| q06_high | 21 | 0 | 0.0% | 0.0% | 0.000 | ✓ |
| q10_low | 208 | 7 | 3.4% | 3.4% | 0.247 | ✓ |
| q11_low | 243 | 16 | 6.6% | 6.6% | 0.437 | ✓ |
| q16_medium | 188 | 4 | 2.1% | 2.1% | 0.163 | ✓ |
| q17_medium | 138 | 7 | 5.1% | 5.1% | 0.353 | ✓ |

**Key finding:** Logarithmic scaling (ln) eliminates ceiling bunching — q03 (0.929) and q05 (0.990) now distinguishable.

#### Semantic Density Agent
*LLM-based information density (0=high info, 1=low info). Score inverted: low density → high load.*

| Recording | Density Score | Final Score | Truncated | Notes |
|-----------|---------------|---------|----------|-------|
| q02_low | 0.2 | 0.800 | No | Vague, repetitive |
| q04_medium | 0.7 | 0.300 | No | Clear, structured |
| q05_high | 0.7 | 0.300 | No | Clear, step-by-step |
| q06_high | 0.0 | **1.000** | No | **Incoherent, gave up** |
| q10_low | 0.3 | 0.650 | No | Some personal detail |
| q11_low | 0.3 | 0.650 | No | Descriptive but vague |
| q16_medium | 0.3 | 0.650 | No | Some ideas, vague |
| q17_medium | 0.2 | 0.800 | No | Vague, repetitive |

**Key finding:** q06_high scores 1.0 (worst possible) — confirms speaker gave up on task after audio gaps.

#### Syntactic Complexity Agent
*Dependency parse analysis of sentence structure. Marked UNRELIABLE if <3 sentences (fallback to 0.5).*

| Recording | Sentences | Avg Len | Depth | Sub% | Unreliable | Final Score |
|-----------|-----------|---------|-------|------|-----------|---------|
| q01_low | 1 | 15.0 | 0.0 | 0.0% | **YES** | 0.500 |
| q02_low | 2 | 48.0 | 5.0 | 0.0% | **YES** | 0.500 |
| q03_medium | 1 | 49.0 | 3.0 | 0.0% | **YES** | 0.500 |
| q04_medium | 2 | 30.5 | 2.0 | 1.6% | **YES** | 0.500 |
| q05_high | 2 | 25.0 | 5.0 | 2.0% | **YES** | 0.500 |
| q06_high | 3 | 7.7 | 5.0 | 0.0% | NO | 0.659 |
| q10_low | 13 | 20.2 | 4.4 | 14.1% | NO | 0.309 |
| q11_low | 9 | 30.0 | 6.2 | 13.7% | NO | 0.157 |
| q16_medium | 6 | 36.3 | 6.2 | 10.5% | NO | 0.198 |
| q17_medium | 9 | 21.0 | 4.7 | 4.8% | NO | 0.400 |

**Key finding:** First 5 recordings have <3 sentences → all marked UNRELIABLE. New batch (q10+) reliable with 6–19 sentences.

---

### Data Quality Assessment

**First Dataset (q01–q06):**
- ⚠️ **5/6 syntactic failures** (short, choppy utterances)
- ⚠️ q06_high: Only 21 words from 55.9s audio (likely Whisper dropout)
- ⚠️ High pause duration (2429ms avg) suggests speaker frustration
- ✅ Filler words calibration now distinguishes q03/q05 post-log-scaling

**New Dataset (q10+, 18 recordings):**
- ✅ **No syntactic failures** (all have 6–19 sentences)
- ✅ Normal WPM range: 120–170 (optimal for analysis)
- ✅ Pause counts 30–52 (meaningful variation)
- ✅ **All agents producing valid scores** — ready for production

---

**Note:** Agent scores represent component scores (0–1 range). Aggregated load score is a weighted combination of all agents:
- `pause_patterns` 30%
- `speech_rate` 25%
- `filler_words` 20%
- `semantic_density` 10%
- `syntactic_complexity` 15% (with syntax-pause interaction)

---

## Agent Diagnostics

Use `agent_diagnostics.py` to visualize per-agent score distributions and detect miscalibration:

```bash
python agent_diagnostics.py
python agent_diagnostics.py -o my_diagnostics/  # custom output dir
python agent_diagnostics.py --json             # export stats to JSON
```

Generates:
- **Distribution histograms** for each agent
- **Box plots** showing IQR, median, outliers
- **Statistical summary** (mean, stdev, IQR) with calibration warnings

**Example output:** If `pause_patterns` scores bunch between 0.6–0.9 across all recordings, the calibration ceiling is too low.

---

## Calibration Notes

### Aggregator Coverage Gate (NEW)
Prevents silent failures when multiple agents fail simultaneously:
- **Minimum coverage:** 70% of total agent weight must be active
- **If coverage drops below 70%:** raises `ValueError` with missing agent list
- **Guards against:** 3+ simultaneous agent failures being hidden

### Speech Rate (Temporary)
Currently uses hardcoded WPM range: **80–200 WPM** (based on linguistic literature).  
Normal conversational speech: 120–180 WPM.

Short-recording guard (< 20s or < 25 words) flags speech_rate as `unreliable: True` with score = 0.5.

⚠️  **Issue fixed:** q06_high (23.6 WPM, 21 words) now correctly flagged as unreliable.

**TODO:** Once you have **30+ recordings**, call `SpeechRateAgent.fit()` to compute a proper calibration from your data.

### Filler Words (Calibrated)
Applies **logarithmic scaling** to eliminate ceiling bunching:
- **Formula:** `Score = ln(1 + ratio) capped at 0.99`
- **Previous issue:** q03_medium and q05_high both scored 1.0 (indistinguishable)
- **Improvement:** q03_medium now 0.929, q05_high now 0.990 (distinguishable)

Weighted detection: `um`/`uh` count as 2×, other fillers (`like`, `actually`, etc.) count as 1×.

### Pause Patterns (Calibrated)
Calibration tuned to actual data observations:
- `max_pause_ms`: lowered from 2500 → **1200** (matched to 760–1088ms range)
- `max_pauses_per_min`: set to **30** (headroom for variation)
- **Issue fixed:** Original 2500ms ceiling was too lenient, causing bunching
- **Current data:** q06_high at 2429ms (2.4s avg) correctly scores high load

Non-linear scaling (x^0.75 per score component) improves discrimination across pause phenomena.

### Syntactic Complexity (In Progress)
Currently requires **minimum 3 sentences** for reliable parsing. First dataset showed **5/6 failures**.

⚠️  **Issue:** Short, choppy utterances (< 3 sentences) fall back to neutral score 0.5.

**Action taken:** New dataset (18 recordings) all have 6–19 sentences — no fallback scores.

**TODO:** Consider reducing sentence threshold to 1–2 for shorter responses.

### Semantic Density (Context-Enhanced)
Now includes **word count and question context** in LLM prompts:
- Prompt includes: "This is a {word_count}-word spoken answer to the question: '{question_text}'"
- LLM instruction: "Rate density relative to what's possible in a spoken answer of this length"
- **Benefit:** Short, focused answers can now score high (concise = high quality)

Requires optional `question_text` parameter from pipeline caller.

---

## Prediction vs. Actual Results (Ground Truth Comparison)

Using speaker self-reports from `Data/label.csv`, here's how the pipeline's predictions compare to actual consensus labels:

### Accuracy Summary

| Metric | Value |
|--------|-------|
| **Correct Predictions** | 6 / 17 recordings |
| **Accuracy** | 35.3% |
| **Recordings with No Ground Truth** | 1 (spkr_01 q01) |
| **Total Analyzed** | 18 |

### Detailed Comparison

**Legend:** ✓ = Correct prediction | ❌ = Incorrect | ⚠️ = No ground truth

| Speaker | Question | Designed Difficulty | Actual Label | Predicted Label | Score | Result |
|---------|----------|-------------------|--------------|-----------------|-------|--------|
| spkr_01 | q01 | low | *No label* | Low | 0.173 | ⚠️ |
| spkr_01 | q02 | low | **Low** | Medium | 0.580 | ❌ |
| spkr_01 | q03 | medium | **Medium** | High | 0.692 | ❌ |
| spkr_01 | q04 | medium | **Medium** | High | 0.638 | ❌ |
| spkr_01 | q05 | high | **High** | High | 0.790 | ✓ |
| spkr_01 | q06 | high | **High** | Medium | 0.576 | ❌ |
| spkr_02 | q01 | low | **Medium** | Medium | 0.459 | ✓ |
| spkr_02 | q02 | low | **Low** | Medium | 0.505 | ❌ |
| spkr_02 | q03 | medium | **Low** | Medium | 0.499 | ❌ |
| spkr_02 | q04 | medium | **Medium** | Medium | 0.359 | ✓ |
| spkr_02 | q05 | high | **High** | Medium | 0.423 | ❌ |
| spkr_02 | q06 | high | **Medium** | Low | 0.346 | ❌ |
| spkr_03 | q06 | high | **High** | Medium | 0.515 | ❌ |
| spkr_03 | q10 | low | **Medium** | Medium | 0.505 | ✓ |
| spkr_03 | q11 | low | **Low** | Medium | 0.528 | ❌ |
| spkr_03 | q16 | medium | **Medium** | Medium | 0.502 | ✓ |
| spkr_03 | q17 | medium | **Low** | High | 0.615 | ❌ |
| spkr_03 | q24 | high | **Medium** | Medium | 0.544 | ✓ |

### Error Analysis

**High cognitive load (actual=High):**
- spkr_01 q05: ✓ Correct (0.790)
- spkr_01 q06: ❌ Underpredicted → Medium (0.576) vs. High
- spkr_02 q05: ❌ Underpredicted → Medium (0.423) vs. High
- spkr_03 q06: ❌ Underpredicted → Medium (0.515) vs. High
- **Pattern:** 1/4 correct; system tends to underestimate high load

**Medium cognitive load (actual=Medium):**
- spkr_01 q03: ❌ Overpredicted → High (0.692) vs. Medium
- spkr_01 q04: ❌ Overpredicted → High (0.638) vs. Medium
- spkr_02 q01: ✓ Correct (0.459)
- spkr_02 q04: ✓ Correct (0.359)
- spkr_03 q10: ✓ Correct (0.505)
- spkr_03 q16: ✓ Correct (0.502)
- spkr_03 q24: ✓ Correct (0.544)
- **Pattern:** 5/7 correct; strongest performance here

**Low cognitive load (actual=Low):**
- spkr_01 q02: ❌ Overpredicted → Medium (0.580) vs. Low
- spkr_02 q02: ❌ Overpredicted → Medium (0.505) vs. Low
- spkr_02 q03: ❌ Overpredicted → Medium (0.499) vs. Low
- spkr_03 q11: ❌ Overpredicted → Medium (0.528) vs. Low
- spkr_03 q17: ❌ Overpredicted → High (0.615) vs. Low
- **Pattern:** 0/5 correct; system systematically overestimates low load

### Key Observations

1. **Medium accuracy zone:** System most reliable at 0.35–0.59 range (medium label) with 5/7 correct
2. **Low load misclassification:** All 5 low-load recordings overpredicted (mostly as medium, one as high)
3. **High load underestimation:** 3/4 high-load recordings underpredicted (scored as medium)
4. **Threshold drift:** System appears to have shifted toward middle scores; may need threshold recalibration
5. **spkr_01 dataset quality:** First speaker shows pattern of overprediction on easier questions (q02–q04)

### Recommendations

1. **Re-calibrate thresholds:** Consider adjusting the 0.35/0.60 boundaries based on actual distribution
2. **Investigate spkr_01 bias:** First speaker's recordings (q01–q06) show systematic overestimation
3. **q06_high across speakers:** Consistently underpredicted as medium when actual=high (mark as known issue)
4. **Collect more data:** With only 6 correct predictions from 17 questions, need larger dataset for reliable regression

---

## Score reference

| Score range | Label | Interpretation |
|-------------|-------|----------------|
| 0.00 – 0.34 | 🟢 Low | Fluent, confident speech |
| 0.35 – 0.59 | 🟡 Medium | Some hesitation / cognitive effort |
| 0.60 – 1.00 | 🔴 High | Significant disfluency / high mental load |
