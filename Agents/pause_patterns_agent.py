"""
PausePatternsAgent
------------------
Input  : word timestamps only  (List[Dict] with 'start', 'end', 'word')
Type   : Algorithmic
Library: pure Python + numpy

Score semantics
---------------
  0.0 → no/very short pauses     (low cognitive load)
  1.0 → frequent long pauses     (high cognitive load)

Three sub-components:
  rate_score     – pauses per minute (normalised)
  duration_score – mean pause length (normalised)
  long_score     – fraction of pauses > long_pause_threshold_ms (extra weight
                   on the notably long hesitation gaps)
"""

import json
import numpy as np
from typing import List, Dict, Optional


class PausePatternsAgent:
    def __init__(
        self,
        threshold_ms: float       = 200.0,   # minimum gap to count as a pause
        max_pauses_per_min: float = 25.0,    # calibration ceiling for rate score
        max_pause_ms: float       = 2500.0,  # calibration ceiling for duration score
        long_pause_threshold_ms: float = 1000.0,  # "notably long" pause cutoff
    ):
        self.threshold_ms            = threshold_ms
        self.max_pauses_per_min      = max_pauses_per_min
        self.max_pause_ms            = max_pause_ms
        self.long_pause_threshold_ms = long_pause_threshold_ms
        self.last_result: Optional[Dict] = None

    # Input validation 

    def _validate_input(self, word_timestamps: List[Dict]) -> None:
        if not isinstance(word_timestamps, list):
            raise ValueError("word_timestamps must be a list")
        for i, w in enumerate(word_timestamps):
            if "start" not in w or "end" not in w:
                raise ValueError(f"Missing 'start'/'end' keys in item {i}")
            if w["end"] < w["start"]:
                raise ValueError(f"Invalid timestamp at index {i}: end < start")

    # Main compute   

    def compute(self, word_timestamps: List[Dict]) -> Dict:
        """
        Parameters
        ----------
        word_timestamps : list of {word, start, end} dicts

        Returns
        -------
        {pause_count, mean_pause_ms, pause_rate_per_min, long_pause_fraction,
         pause_durations_ms, score}
        """
        self._validate_input(word_timestamps)

        if len(word_timestamps) < 2:
            result = {
                "pause_count":          0,
                "mean_pause_ms":        0.0,
                "pause_rate_per_min":   0.0,
                "long_pause_fraction":  0.0,
                "pause_durations_ms":   [],
                "score":                0.5,
            }
            self.last_result = result
            return result

        # Collect all inter-word gaps that exceed threshold
        gaps: List[float] = []
        for i in range(1, len(word_timestamps)):
            gap_ms = (word_timestamps[i]["start"] - word_timestamps[i - 1]["end"]) * 1000.0
            if gap_ms > self.threshold_ms:
                gaps.append(gap_ms)

        duration_sec = word_timestamps[-1]["end"] - word_timestamps[0]["start"]
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

        pause_rate  = len(gaps) / duration_min if duration_min > 0 else 0.0
        mean_pause  = float(np.mean(gaps)) if gaps else 0.0
        long_frac   = (
            sum(1 for g in gaps if g > self.long_pause_threshold_ms) / len(gaps)
            if gaps else 0.0
        )

        # Score components 
        rate_score     = min(pause_rate / self.max_pauses_per_min, 1.0)
        duration_score = min(mean_pause  / self.max_pause_ms,      1.0)
        long_score     = long_frac                                        # 0–1 already

        # Weighted blend: rate is most important; long pauses get bonus weight
        score = round(rate_score * 0.45 + duration_score * 0.35 + long_score * 0.20, 3)

        result = {
            "pause_count":          len(gaps),
            "mean_pause_ms":        round(mean_pause, 1),
            "pause_rate_per_min":   round(pause_rate, 2),
            "long_pause_fraction":  round(long_frac,  3),
            "pause_durations_ms":   [round(g, 1) for g in gaps],
            "score":                score,
        }
        self.last_result = result
        return result

    # Persistence 

    def save(self, path: str) -> None:
        data = {
            "threshold_ms":            self.threshold_ms,
            "max_pauses_per_min":      self.max_pauses_per_min,
            "max_pause_ms":            self.max_pause_ms,
            "long_pause_threshold_ms": self.long_pause_threshold_ms,
            "last_result":             self.last_result,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.threshold_ms            = data["threshold_ms"]
        self.max_pauses_per_min      = data["max_pauses_per_min"]
        self.max_pause_ms            = data["max_pause_ms"]
        self.long_pause_threshold_ms = data.get("long_pause_threshold_ms", 1000.0)
        self.last_result             = data.get("last_result")


# Standalone runner 

def main():
    from pathlib import Path

    DATA_DIR   = Path("baby-data/transcripts")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "pause_metrics.json"

    TRANSCRIPTS = {
        "low":    DATA_DIR / "low_transcript.json",
        "medium": DATA_DIR / "medium_transcript.json",
        "high":   DATA_DIR / "high_transcript.json",
    }

    agent = PausePatternsAgent(
        threshold_ms=200,
        max_pauses_per_min=25,
        max_pause_ms=2500,
        long_pause_threshold_ms=1000,
    )

    print("=== Pause Patterns Analysis ===\n")

    for label, path in TRANSCRIPTS.items():
        data            = json.load(open(path))
        word_timestamps = data["words"]

        result = agent.compute(word_timestamps)
        print(f"  [{label.upper()}]")
        print(f"    pause_count          : {result['pause_count']}")
        print(f"    mean_pause_ms        : {result['mean_pause_ms']} ms")
        print(f"    pause_rate_per_min   : {result['pause_rate_per_min']}")
        print(f"    long_pause_fraction  : {result['long_pause_fraction']:.0%}")
        print(f"    score                : {result['score']}  (0 = fluent, 1 = very hesitant)")
        print()

    agent.save(SAVE_PATH)
    print(f"Saved config + last result → {SAVE_PATH}")

    agent2 = PausePatternsAgent()
    agent2.load(SAVE_PATH)
    print(f"Re-loaded threshold  : {agent2.threshold_ms} ms")
    print(f"Re-loaded last_result: {agent2.last_result}")


if __name__ == "__main__":
    main()