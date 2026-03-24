"""
aggregator_agent.py
-------------------
Combines scores from all sub-agents into a single cognitive-load score.
"""

DEFAULT_WEIGHTS = {
    "speech_rate": 0.30,
    "pause_patterns": 0.35,
    "filler_words": 0.25,
    "semantic_density": 0.10,
}

_LOW_THRESHOLD = 0.35
_HIGH_THRESHOLD = 0.60


def aggregator(agent_scores: dict, weights: dict = None) -> dict:
    w = weights or DEFAULT_WEIGHTS

    active_keys = [k for k in w if k in agent_scores]
    if not active_keys:
        raise ValueError("No valid agent scores provided.")

    total_weight = sum(w[k] for k in active_keys)
    norm_w = {k: w[k] / total_weight for k in active_keys}

    contributions = {
        k: round(agent_scores[k]["score"] * norm_w[k], 4)
        for k in active_keys
    }

    load_score = round(sum(contributions.values()), 4)

    if load_score < _LOW_THRESHOLD:
        load_label = "low"
    elif load_score < _HIGH_THRESHOLD:
        load_label = "medium"
    else:
        load_label = "high"

    return {
        "load_score": load_score,
        "load_label": load_label,
        "weights": norm_w,
        "contributions": contributions,
    }