"""
aggregator_agent.py
-------------------
Combines scores from all sub-agents into a single cognitive-load score
and attaches a confidence score computed from data quality signals.

Core agents and weights (simple additive model)
-----------------------------------------------
  pause_patterns:   0.45   hesitation gaps
  filler_words:     0.35   disfluency markers
  semantic_density: 0.20   LLM-rated conceptual richness

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
    "semantic_density": 0.20,  # Supplementary
}

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

    Returns
    -------
    {
        load_score,  # z-score (consumer applies their own thresholds)
        weights, contributions,
        confidence,  # 0.0–1.0 (consumer applies their own thresholds)
        penalties, diagnostics,
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

    # Coverage gate: require 80% of weight
    total_weight = sum(w.values())
    active_weight = sum(w[k] for k in active_keys)
    weight_coverage = active_weight / total_weight if total_weight > 0 else 0.0
    
    if weight_coverage < MIN_WEIGHT_COVERAGE:
        missing = [k for k in w if k not in active_keys]
        raise ValueError(
            f"Only {weight_coverage:.0%} of weight covered (need {MIN_WEIGHT_COVERAGE:.0%}). "
            f"Missing: {missing}"
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

    return {
        "load_score":       load_score,
        "weights":          norm_w,
        "contributions":    contributions,
        "confidence":       conf["confidence"],
        "penalties":        conf["penalties"],
        "diagnostics":      conf["diagnostics"],
    }