"""
confidence_agent.py
-------------------
Computes a confidence score for the aggregated cognitive-load estimate.

Confidence is LOW when:
  1. Audio / transcript is very short            (not enough data)
  2. Transcript is noisy                         (garbled ASR, random chars)
  3. Agent signals disagree with each other      (conflicting evidence)
  4. Semantic agent fell back / was truncated    (LLM signal unreliable)
  5. Topic coherence is poor                     (embedding-free; vocab-window overlap)

Score range
-----------
  0.0  → no confidence  (result is essentially a guess)
  1.0  → high confidence (all signals agree, clean transcript, enough data)
"""

import re
import math
import numpy as np
from typing import List, Dict, Optional


# Thresholds / calibration constants

MIN_WORDS_HIGH_CONF   = 80    # below this → penalise for shortness
MIN_WORDS_ANY_SIGNAL  = 15    # below this → very low floor
MIN_DURATION_SEC_HIGH = 30.0  # audio shorter than this → extra penalty
MIN_DURATION_SEC_ANY  = 5.0   # below this → confidence capped at 0.3

NOISE_ALPHA_CHAR      = 0.15  # fraction of "weird" chars that scores max noise penalty
MAX_WORD_LENGTH       = 25    # words longer than this are likely garbled
GARBLED_WORD_THRESH   = 0.05  # fraction of garbled words that hits max noise penalty

TOPIC_WINDOW_WORDS    = 40    # sliding window size (words) for coherence
TOPIC_STEP_WORDS      = 20    # step between windows
MIN_TOPIC_OVERLAP     = 0.10  # Jaccard below this → low coherence
LOW_COHERENCE_FRAC    = 0.50  # fraction of window pairs that must fail to penalise

SIGNAL_SPREAD_HIGH    = 0.40  # std-dev of agent scores above this → low agreement


def _word_list(text: str) -> List[str]:
    """Lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def _jaccard(set_a: set, set_b: set) -> float:
    """Vocabulary-overlap Jaccard similarity."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# Individual penalty functions (each returns a value in [0, 1];
# 0 = no penalty, 1 = maximum penalty)

def _duration_penalty(words: List[Dict], text: str) -> tuple:
    """Penalty for very short recordings."""
    n_words = len(words) if words else len(_word_list(text))
    duration_sec = 0.0
    if words and len(words) >= 2:
        duration_sec = words[-1]["end"] - words[0]["start"]

    word_penalty = 0.0
    if n_words < MIN_WORDS_ANY_SIGNAL:
        word_penalty = 1.0
    elif n_words < MIN_WORDS_HIGH_CONF:
        word_penalty = 1.0 - (n_words - MIN_WORDS_ANY_SIGNAL) / (
            MIN_WORDS_HIGH_CONF - MIN_WORDS_ANY_SIGNAL
        )

    dur_penalty = 0.0
    if duration_sec > 0:
        if duration_sec < MIN_DURATION_SEC_ANY:
            dur_penalty = 1.0
        elif duration_sec < MIN_DURATION_SEC_HIGH:
            dur_penalty = 1.0 - (duration_sec - MIN_DURATION_SEC_ANY) / (
                MIN_DURATION_SEC_HIGH - MIN_DURATION_SEC_ANY
            )

    return max(word_penalty, dur_penalty), n_words, duration_sec


def _noise_penalty(text: str) -> float:
    """Penalty for a garbled / noisy transcript."""
    if not text or not text.strip():
        return 1.0

    # Fraction of non-ASCII or non-letter/space/common-punct chars
    weird = sum(
        1 for c in text
        if not (c.isascii() and (c.isalpha() or c in " ,.?!'-\n"))
    )
    char_noise = min(weird / max(len(text), 1), NOISE_ALPHA_CHAR) / NOISE_ALPHA_CHAR

    # Fraction of "garbled" words (very long or all-digit)
    words = _word_list(text)
    if not words:
        return 1.0
    garbled = sum(1 for w in words if len(w) > MAX_WORD_LENGTH or w.isdigit())
    garbled_frac = min(garbled / len(words), GARBLED_WORD_THRESH) / GARBLED_WORD_THRESH

    return max(char_noise, garbled_frac)


def _agreement_penalty(agent_scores: Dict[str, Dict]) -> float:
    """Penalty for high disagreement between agent scores."""
    scores = [v["score"] for v in agent_scores.values() if "score" in v]
    if len(scores) < 2:
        return 0.5  # can't judge
    std = float(np.std(scores))
    return min(std / SIGNAL_SPREAD_HIGH, 1.0)


def _semantic_reliability_penalty(semantic_result: Dict) -> float:
    """Penalty when the LLM semantic agent was unreliable."""
    penalty = 0.0
    if semantic_result.get("truncated"):
        penalty += 0.3
    reasoning = semantic_result.get("reasoning", "")
    # Fallback message baked into the agent
    if "fallback" in reasoning.lower() or "did not return" in reasoning.lower():
        penalty += 0.5
    # If score is exactly 0.5 it's suspicious (LLM default)
    if abs(semantic_result.get("score", -1) - 0.5) < 1e-6:
        penalty += 0.1
    return min(penalty, 1.0)


def _topic_coherence_penalty(text: str) -> float:
    """
    Embedding-free topic-transition penalty.

    Splits text into overlapping vocabulary windows and measures
    Jaccard overlap between adjacent windows.  A high fraction of
    low-overlap pairs → speech is incoherent / topic-scattered.
    """
    tokens = _word_list(text)
    if len(tokens) < TOPIC_WINDOW_WORDS * 2:
        return 0.0  # not enough tokens to judge

    # Build vocabulary sets for each window
    windows = []
    i = 0
    while i + TOPIC_WINDOW_WORDS <= len(tokens):
        windows.append(set(tokens[i: i + TOPIC_WINDOW_WORDS]))
        i += TOPIC_STEP_WORDS

    if len(windows) < 2:
        return 0.0

    low_overlap_count = 0
    for j in range(len(windows) - 1):
        jac = _jaccard(windows[j], windows[j + 1])
        if jac < MIN_TOPIC_OVERLAP:
            low_overlap_count += 1

    low_frac = low_overlap_count / (len(windows) - 1)
    return min(low_frac / LOW_COHERENCE_FRAC, 1.0)


# Main entry point

def compute_confidence(
    words: List[Dict],
    text: str,
    agent_scores: Dict[str, Dict],
    *,
    # weights for each penalty component (must sum to 1.0)
    w_duration:    float = 0.25,
    w_noise:       float = 0.20,
    w_agreement:   float = 0.25,
    w_semantic:    float = 0.15,
    w_coherence:   float = 0.15,
) -> Dict:
    """
    Compute a confidence score for the cognitive-load estimate.

    Parameters
    ----------
    words        : list of {word, start, end} dicts from Whisper
    text         : raw transcript string
    agent_scores : dict of agent name → agent result dict (must contain 'score')

    Returns
    -------
    {
        "confidence":        float,      # 0.0–1.0
        "confidence_label":  str,        # "low" / "medium" / "high"
        "penalties": {
            "short_audio":       float,
            "noisy_transcript":  float,
            "signal_disagreement": float,
            "semantic_unreliable": float,
            "topic_incoherence": float,
        },
        "diagnostics": {
            "word_count":    int,
            "duration_sec":  float,
            "score_spread":  float,
        }
    }
    """
    # 1. Per-component penalties
    semantic_result = agent_scores.get("semantic_density", {})

    dur_pen, n_words, dur_sec = _duration_penalty(words, text)
    noise_pen      = _noise_penalty(text)
    agree_pen      = _agreement_penalty(agent_scores)
    sem_pen        = _semantic_reliability_penalty(semantic_result)
    coh_pen        = _topic_coherence_penalty(text)

    # 2. Weighted penalty → confidence
    total_penalty = (
        w_duration  * dur_pen   +
        w_noise     * noise_pen +
        w_agreement * agree_pen +
        w_semantic  * sem_pen   +
        w_coherence * coh_pen
    )

    confidence = round(max(0.0, 1.0 - total_penalty), 3)

    # Hard cap: if audio is extremely short, cap confidence
    if n_words < MIN_WORDS_ANY_SIGNAL or dur_sec < MIN_DURATION_SEC_ANY:
        confidence = min(confidence, 0.30)

    if confidence < 0.40:
        label = "low"
    elif confidence < 0.70:
        label = "medium"
    else:
        label = "high"

    agent_score_values = [v["score"] for v in agent_scores.values() if "score" in v]
    spread = round(float(np.std(agent_score_values)), 4) if len(agent_score_values) >= 2 else 0.0

    return {
        "confidence": confidence,
        "confidence_label": label,
        "penalties": {
            "short_audio":          round(dur_pen,   3),
            "noisy_transcript":     round(noise_pen, 3),
            "signal_disagreement":  round(agree_pen, 3),
            "semantic_unreliable":  round(sem_pen,   3),
            "topic_incoherence":    round(coh_pen,   3),
        },
        "diagnostics": {
            "word_count":   n_words,
            "duration_sec": round(dur_sec, 1),
            "score_spread": spread,
        },
    }
