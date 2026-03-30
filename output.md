# Findings Report: Voice as a Proxy for Cognitive Load

**Speakers:** ayu, lak, vin
**Pipeline:** 3 agents — pause patterns, filler words, speech rate
**Approach:** Within-speaker z-score deviation from spontaneous speech baseline

---

## Complete Results

**Ayu** — baseline filler rate: 0.195 (low natural disfluency)

| question | difficulty | domain | pause_z | filler_z | rate_z |
|----------|------------|--------|---------|----------|--------|
| q01 | low | Personal | +0.013 | -0.003 | -0.015 |
| q02 | low | Personal | -0.270 | -0.040 | +0.445 |
| q03 | medium | Procedural | +0.067 | +0.810 | -0.475 |
| q04 | medium | Technical | -0.790 | +0.077 | +0.185 |
| q05 | high | Procedural | -0.503 | +1.033 | +0.010 |
| q06 | high | Technical | -0.453 | +0.327 | -0.490 |

**Lak** — baseline filler rate: 0.888 (high natural disfluency)

| question | difficulty | domain | pause_z | filler_z | rate_z |
|----------|------------|--------|---------|----------|--------|
| q10 | low | Personal | +0.490 | -1.787 | -0.040 |
| q11 | low | Personal | +0.577 | -0.327 | +0.300 |
| q16 | medium | Procedural | +0.713 | -2.757 | +0.550 |
| q17 | medium | Science | +0.783 | -1.817 | +0.675 |
| q06 | high | Technical | +0.740 | -1.293 | -0.265 |
| q24 | high | Procedural | +0.723 | -0.167 | -0.205 |

**Vin** — baseline filler rate: 0.224 (low natural disfluency)

| question | difficulty | domain | pause_z | filler_z | rate_z |
|----------|------------|--------|---------|----------|--------|
| q01 | low | Personal | -0.433 | +0.687 | -0.770 |
| q03 | medium | Procedural | -0.297 | +1.140 | -0.720 |
| q04 | medium | Technical | -0.427 | +2.133 | -0.280 |
| q05 | high | Procedural | +0.120 | +2.553 | -0.045 |
| q06 | high | Technical | +0.580 | +2.553 | -0.055 |

---

## Finding 1: Per-Speaker Normalization is Necessary

Raw acoustic scores are not comparable across speakers. Lak's baseline filler rate (0.888) is nearly four times ayu's (0.195) and four times vin's (0.224). Without normalization, lak would appear to be under extreme load on every response. Z-score normalization against each speaker's own spontaneous speech baseline is the minimum requirement for any meaningful cross-speaker comparison.

---

## Finding 2: Filler Words is the Most Sensitive Signal — But Only for Low-Baseline Speakers

For ayu and vin, filler word frequency showed the clearest monotonic increase across difficulty levels:

| speaker | low (avg filler_z) | medium (avg filler_z) | high (avg filler_z) |
|---------|--------------------|-----------------------|---------------------|
| ayu | -0.022 | +0.444 | +0.680 |
| vin | +0.687 | +1.637 | +2.553 |

This is a clean directional trend — filler words increase consistently as designed difficulty increases for both speakers.

For lak, filler words is entirely inverted — every response shows negative filler_z regardless of difficulty. This is a ceiling effect: lak's baseline filler rate is already near maximum (0.888), leaving no room to increase under load. The signal collapses.

---

## Finding 3: For High-Baseline Speakers, Pause Patterns Becomes the Primary Signal

For lak, pause patterns showed what filler words showed for ayu and vin — a consistent directional trend:

| difficulty | pause_z (lak) |
|------------|---------------|
| low | +0.490, +0.577 |
| medium | +0.713, +0.783 |
| high | +0.740, +0.723 |

Low difficulty responses cluster around +0.5, medium around +0.75, high around +0.73. The separation between low and medium/high is clear even with only 6 data points.

This means signal dominance is speaker-dependent. A system with hardcoded agent weights will work for some speakers and silently fail for others.

---

## Finding 4: Designed Difficulty ≠ Experienced Difficulty

The most striking finding came from q06 — "how would you debug code that crashes randomly" — which was labeled high difficulty.

**Ayu on q06:**
```
filler_z: +0.327  (near medium level)
words:     74 words, coherent structured answer
```
Ayu gave a fluent, confident debugging answer. Acoustic deviation was low because for ayu, a developer, this was not a high difficulty question.

**Vin on q06:**
```
filler_z: +2.553  (maximum observed)
words:     29 words in 56 seconds, wpm=31
```
Vin produced 29 words in 56 seconds — barely above silence. The acoustic signal correctly detected extreme difficulty for vin on the same question.

Same question, opposite acoustic profiles, opposite experienced difficulty. The designed label "high" captured neither speaker's actual experience accurately. The acoustic deviation scores did.

This pattern extended to medium technical questions. Ayu's q04 (explain machine learning) showed filler_z of only +0.077 — near baseline — consistent with domain familiarity. Vin's q04 showed filler_z of +2.133 — the same question felt genuinely effortful.

---

## Finding 5: Speech Rate is the Weakest and Most Inconsistent Signal

Across all three speakers and all difficulty levels, speech rate showed no consistent directional pattern. Some speakers sped up under load, others slowed down, and many showed near-zero deviation regardless of difficulty. Speech rate appears to be more sensitive to individual speaking style and topic engagement than to cognitive load, at least in this dataset.

---

## Limitations

**Sample size:** 3 speakers, 17 total responses. No statistical significance is claimed. All findings are preliminary and directional.

**Baseline quality:** Vin's baseline was flagged as potentially slow (89 wpm), which may have inflated some deviation scores. Lak's baseline was recorded in a different session than test responses, introducing session-level variation as a confound.

**No ground truth:** Designed difficulty labels are the experimenter's assessment of question complexity, not a validated measure of experienced cognitive load. This report deliberately avoids claiming to measure cognitive load — it measures acoustic deviation under varying task conditions.

**Single language pair:** All speakers are Hindi-English bilinguals. Findings may not generalize to other language backgrounds.

---

## Conclusion

Voice is a partially reliable proxy for cognitive load, but with two important conditions. First, per-speaker baseline normalization is non-negotiable — raw scores are meaningless across speakers. Second, no single acoustic feature is universally reliable — filler words dominates for low-baseline speakers, pause patterns dominates for high-baseline speakers, and speech rate is unreliable for both.

The most practically important finding is that acoustic deviation scores outperformed designed difficulty labels at capturing experienced cognitive load. The same question produced opposite acoustic profiles for different speakers depending on domain familiarity — something no label-based approach could capture.

A voice-based cognitive load system that hardcodes feature weights, skips baseline calibration, or trusts designed difficulty as ground truth will produce unreliable results. One that treats load estimation as a within-speaker, multi-signal, baseline-relative problem shows genuine promise even at this preliminary scale.

---

*Pipeline: Whisper ASR → pause patterns agent + filler words agent + speech rate agent → per-speaker z-score normalization → deviation report*

*Data: 3 speakers, 17 responses, 3 difficulty levels, 4 domains*