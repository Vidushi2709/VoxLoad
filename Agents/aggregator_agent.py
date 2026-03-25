"""
aggregator_agent.py
-------------------
Combines scores from all sub-agents into a single cognitive-load score
and attaches a confidence score computed from data quality signals.

Agents and default weights
--------------------------
  speech_rate          0.25   how fast / how variable
  pause_patterns       0.30   hesitation gaps  (nonlinear-scaled in the agent)
  filler_words         0.20   disfluency markers
  semantic_density     0.10   LLM-rated conceptual richness
  syntactic_complexity 0.15   sentence depth / clause structure

Syntax × Pause Interaction
--------------------------
  Normal (no pauses): high complexity → low load score ✓  (expert speaker)
  High pauses + high complexity: load stays high          (struggling with
      complex material — pauses are NOT topic transitions here)

  As pause_score rises above PAUSE_INTERACTION_FLOOR, the "expert discount"
  that syntactic complexity provides is linearly cancelled.  At pause_score ≥
  PAUSE_INTERACTION_CEIL the discount is fully removed.

Confidence:
  Derived from: audio length, transcript noise, agent agreement,
  semantic reliability, and vocab-window coherence.
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
    "speech_rate":          0.25,
    "pause_patterns":       0.30,
    "filler_words":         0.20,
    "semantic_density":     0.10,
    "syntactic_complexity": 0.15,
}

_LOW_THRESHOLD  = 0.35
_HIGH_THRESHOLD = 0.60

# Interaction thresholds
# Below FLOOR  → full syntax expert discount applied
# Above CEIL   → syntax discount fully cancelled (pauses dominate)
PAUSE_INTERACTION_FLOOR = 0.35
PAUSE_INTERACTION_CEIL  = 0.75

# Weight coverage gate: prevent silent failures from too many agent failures
MIN_WEIGHT_COVERAGE = 0.70  # 70% of total weight must be active


def _apply_interaction(agent_scores: dict, contributions: dict, norm_w: dict) -> tuple:
    """
    Syntax × Pause interaction correction.

    When pause_score is high AND syntax score is low (= high complexity),
    the system would normally reduce load because complex syntax signals
    an expert speaker.  But high pauses + complex syntax means the speaker
    is genuinely struggling with the material — the load stays high.

    Returns (adjusted_contributions, interaction_delta) where
    interaction_delta > 0 means pauses overrode the syntax discount.
    """
    if "syntactic_complexity" not in agent_scores or "pause_patterns" not in agent_scores:
        return contributions, 0.0

    pause_s  = agent_scores["pause_patterns"]["score"]
    syntax_s = agent_scores["syntactic_complexity"]["score"]

    expert_headroom = max(0.0, 0.5 - syntax_s)          # 0 if already high-load
    if expert_headroom < 1e-4 or "syntactic_complexity" not in norm_w:
        return contributions, 0.0

    adj = dict(contributions)
    interaction_delta = 0.0

    # High pause: cancel the syntax discount
    factor = (pause_s - PAUSE_INTERACTION_FLOOR) / (
        PAUSE_INTERACTION_CEIL - PAUSE_INTERACTION_FLOOR
    )
    factor = max(0.0, min(1.0, factor))

    if factor >= 1e-4:
        discount_cancelled = norm_w["syntactic_complexity"] * expert_headroom * factor
        adj["syntactic_complexity"] = round(
            contributions["syntactic_complexity"] + discount_cancelled, 4
        )
        interaction_delta = round(discount_cancelled, 4)
    # New: Low pause + high complexity → apply expert discount explicitly
    elif pause_s < PAUSE_INTERACTION_FLOOR and syntax_s < 0.3:
        # Speaker is fluent AND syntactically complex → reduce load score
        expert_bonus = norm_w["syntactic_complexity"] * (0.3 - syntax_s) * (
            1 - pause_s / PAUSE_INTERACTION_FLOOR
        )
        adj["syntactic_complexity"] = max(0.0, round(
            contributions["syntactic_complexity"] - expert_bonus, 4
        ))
        interaction_delta = round(-expert_bonus, 4)
    else:
        # Neutral zone: moderate pauses and/or moderate complexity
        # No interaction effect applies; contributions used as-is
        # interaction_delta remains 0.0 (no adjustment)
        pass

    return adj, interaction_delta


def aggregator(
    agent_scores: dict,
    weights: dict = None,
    *,
    words: Optional[List[Dict]] = None,
    text:  Optional[str]        = None,
) -> dict:
    """
    Parameters
    ----------
    agent_scores : dict of agent name → agent result dict (must contain 'score')
    weights      : optional weight override (missing keys are silently ignored)
    words        : Whisper word-timestamp list  (used by confidence scorer)
    text         : raw transcript string        (used by confidence scorer)

    Returns
    -------
    {
        load_score, load_label,
        weights, contributions,
        interaction_delta,
        confidence, confidence_label,
        penalties, diagnostics,
    }
    """
    w = weights or DEFAULT_WEIGHTS

    # Only score agents that actually ran
    active_keys = [k for k in w if k in agent_scores]
    
    # Exclude unreliable agents and redistribute their weight
    active_keys = [
        k for k in active_keys
        if not agent_scores[k].get("unreliable", False)
    ]
    
    if not active_keys:
        raise ValueError("No valid agent scores provided.")

    # Coverage gate: ensure enough agent weight is active
    total_weight = sum(w.values())  # Sum of ALL weights
    active_weight = sum(w[k] for k in active_keys)
    weight_coverage = active_weight / total_weight if total_weight > 0 else 0.0
    
    if weight_coverage < MIN_WEIGHT_COVERAGE:
        missing_agents = [k for k in w if k not in agent_scores or agent_scores[k].get("unreliable", False)]
        raise ValueError(
            f"Only {weight_coverage:.0%} of total weight covered (need {MIN_WEIGHT_COVERAGE:.0%}). "
            f"Missing/unreliable agents: {missing_agents}"
        )

    total_weight = active_weight  # Use active weight for normalization
    norm_w = {k: w[k] / total_weight for k in active_keys}

    contributions = {
        k: round(agent_scores[k]["score"] * norm_w[k], 4)
        for k in active_keys
    }

    # Apply syntax × pause interaction before summing
    contributions, interaction_delta = _apply_interaction(
        agent_scores, contributions, norm_w
    )

    load_score = round(sum(contributions.values()), 4)

    if load_score < _LOW_THRESHOLD:
        load_label = "low"
    elif load_score < _HIGH_THRESHOLD:
        load_label = "medium"
    else:
        load_label = "high"

    # Confidence score 
    conf = compute_confidence(
        words or [],
        text  or "",
        agent_scores,
    )

    return {
        "load_score":         load_score,
        "load_label":         load_label,
        "weights":            norm_w,
        "contributions":      contributions,
        "interaction_delta":  interaction_delta,   # > 0 means pauses overrode syntax discount
        "confidence":         conf["confidence"],
        "confidence_label":   conf["confidence_label"],
        "penalties":          conf["penalties"],
        "diagnostics":        conf["diagnostics"],
    }