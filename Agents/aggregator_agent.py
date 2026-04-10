"""
aggregator_agent.py
-------------------
Combines scores from all sub-agents into a single cognitive-load score
and attaches a confidence score computed from data quality signals.

Core agents and weights (simple additive model)
-----------------------------------------------
  filler_words:     0.50   disfluency markers          (strongest discriminator)
  pause_patterns:   0.30   hesitation gaps
  speech_rate:      0.20   speaking pace

Optional supplementary agent (activates when present)
------------------------------------------------------
  coherence:        0.15   LLM-rated content coherence  (orthogonal to acoustics)

  When coherence is present, weights are re-normalised automatically so
  the coverage gate never rejects a run that includes it.

Confidence:
  Derived from: audio length, transcript noise, and agent reliability.
"""

import sys
import os
from typing import List, Dict, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Agents.confidence_score import compute_confidence   


DEFAULT_WEIGHTS = {
    "filler_words":     0.50,  # STRONGEST discriminator (0.362→0.686 across classes)
    "pause_patterns":   0.30,  # Secondary signal
    "speech_rate":      0.20,  # Pace / fluency signal
    # Optional — only counted when present; weights re-normalised automatically
    "coherence":        0.15,  # LLM content coherence (orthogonal to acoustics)
}

# Agents required for the pipeline to proceed (coherence is optional)
REQUIRED_AGENTS = {"filler_words", "pause_patterns", "speech_rate"}

# Weight coverage gate: require 80% of weight to be active
MIN_WEIGHT_COVERAGE = 0.80

def aggregator(
    agent_scores: dict,
    weights: dict = None,
    *,
    words: Optional[List[Dict]] = None,
    text:  Optional[str]        = None,
) -> dict:
    """
    Combine agent scores using simple additive weighted model.

    Parameters
    ----------
    agent_scores : dict of agent name → agent result dict (must contain 'score')
    weights      : optional weight override (defaults to DEFAULT_WEIGHTS)
    words        : Whisper word-timestamp list (used for confidence scoring)
    text         : raw transcript string (used for confidence scoring)

    Notes
    -----
    coherence is optional — if absent it is silently excluded and weights are
    re-normalised over the remaining active agents.  The coverage gate only
    demands that REQUIRED_AGENTS are all present.

    Returns
    -------
    {
        load_score,  # weighted z-score composite
        weights, contributions,
        confidence,  # 0.0–1.0
        penalties, diagnostics,
        coherence_used,  # bool — whether the coherence agent contributed
    }
    """
    w = weights or DEFAULT_WEIGHTS

    # Only use agents that ran and are not marked unreliable
    active_keys = [
        k for k in w
        if k in agent_scores and not agent_scores[k].get("unreliable", False)
    ]

    if not active_keys:
        raise ValueError("No valid agent scores provided.")

    # Verify all required agents are present
    missing_required = REQUIRED_AGENTS - set(active_keys)
    if missing_required:
        raise ValueError(
            f"Required agents missing: {missing_required}. "
            f"Cannot compute reliable load score."
        )

    # Coverage gate: compute over required agents only (coherence is optional)
    required_weight  = sum(w.get(k, 0) for k in REQUIRED_AGENTS)
    active_req_weight = sum(w.get(k, 0) for k in REQUIRED_AGENTS if k in active_keys)
    weight_coverage  = active_req_weight / required_weight if required_weight > 0 else 0.0

    if weight_coverage < MIN_WEIGHT_COVERAGE:
        raise ValueError(
            f"Only {weight_coverage:.0%} of required-agent weight covered "
            f"(need {MIN_WEIGHT_COVERAGE:.0%}).  Missing required: {missing_required}"
        )

    # Normalize weights
    norm_w = {k: w[k] / active_weight for k in active_keys}

    # Simple additive scoring
    contributions = {
        k: round(agent_scores[k]["score"] * norm_w[k], 4)
        for k in active_keys
    }

    load_score = round(sum(contributions.values()), 4)

    # Confidence score
    conf = compute_confidence(
        words or [],
        text  or "",
        agent_scores,
    )

    coherence_used = "coherence" in active_keys

    return {
        "load_score":       load_score,
        "weights":          norm_w,
        "contributions":    contributions,
        "confidence":       conf["confidence"],
        "penalties":        conf["penalties"],
        "diagnostics":      conf["diagnostics"],
        "coherence_used":   coherence_used,
    }