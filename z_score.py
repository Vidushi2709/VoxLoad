"""
z_score.py
----------
Z-score normalization utilities for agents.

With only 1-3 speakers, we can't compute population std yet.
Instead, use expected_range which represents typical within-speaker variation.
"""

# Z-score normalization: Expected within-speaker variation range per feature
FEATURE_EXPECTED_RANGE = {
    "filler_words":     0.3,   # typical within-speaker variation
    "pause_patterns":   0.3,
    "speech_rate":      0.2,
    "semantic_density": 0.3,
}


def compute_z_score(raw_score: float, baseline: float, feature_name: str) -> float:
    """
    Compute z-score relative to speaker baseline using expected within-speaker range.
    
    With only 1-3 speakers, we can't compute population std yet.
    Instead, use expected_range which represents typical variation from baseline.
    
    Parameters
    ----------
    raw_score : float
        Raw agent score (0.0-1.0 typically)
    baseline : float
        Speaker's baseline score for this agent
    feature_name : str
        Agent name ("filler_words", "pause_patterns", etc.)
    
    Returns
    -------
    float
        Z-score: (raw_score - baseline) / expected_range
    """
    expected_range = FEATURE_EXPECTED_RANGE.get(feature_name, 0.3)
    return round((raw_score - baseline) / expected_range, 4)
