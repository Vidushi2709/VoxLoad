"""
FillerPatternsAgent
-------------------
Input  : transcript text only  (plain string)
Type   : Algorithmic
Library: regex / spaCy  (regex used by default; spaCy used for lemmatised matching)

Score semantics
---------------
  0.0 → no filler words           (low cognitive load)
  1.0 → filler-heavy speech       (high cognitive load)

Two weighting tiers
-------------------
  HIGH-weight fillers  – strong disfluency markers:  "um", "uh", "er", "erm"
  NORMAL fillers       – hedges / discourse markers:  "like", "you know", etc.

High-weight fillers contribute 2× to the total filler count before rate scoring.
"""

import re
import json
from typing import Dict, List, Optional, Set


try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# Filler word taxonomy 

# Strong disfluency markers — counted at 2× weight
HIGH_WEIGHT_FILLERS: Set[str] = {
    "um", "uh", "er", "erm", "uhh", "umm",
}

# Hedges, discourse markers, and verbal tics — counted at 1× weight
NORMAL_FILLERS: Set[str] = {
    "like", "you know", "basically", "literally",
    "kind of", "sort of", "i mean", "right",
    "actually", "honestly", "obviously", "clearly",
    "well", "okay", "so", "anyway",
}

ALL_FILLERS: Set[str] = HIGH_WEIGHT_FILLERS | NORMAL_FILLERS


class FillerPatternsAgent:
    def __init__(
        self,
        fillers:          Optional[Set[str]] = None,
        high_weight:      Optional[Set[str]] = None,
        max_filler_rate:  float              = 0.12,  # weighted fillers / total words
        high_weight_mult: float              = 2.0,   # multiplier for strong disfluency
    ):
        self.fillers          = fillers      or ALL_FILLERS
        self.high_weight      = high_weight  or HIGH_WEIGHT_FILLERS
        self.max_filler_rate  = max_filler_rate
        self.high_weight_mult = high_weight_mult
        self._patterns        = self._compile(self.fillers)
        self.last_result: Optional[Dict] = None

    # Pattern compilation 

    @staticmethod
    def _compile(fillers: Set[str]) -> Dict[str, re.Pattern]:
        patterns = {}
        for filler in fillers:
            # Use word boundaries; for single words like "so" require real boundaries
            patterns[filler] = re.compile(
                r"(?<!\w)" + re.escape(filler) + r"(?!\w)",
                re.IGNORECASE,
            )
        return patterns

    # Counting methods 

    def _count_with_spacy(self, text: str) -> Dict[str, int]:
        """
        Use spaCy lemmatisation so case variants and mild inflections are caught.
        Falls back to regex if spaCy is unavailable.
        """
        if not SPACY_AVAILABLE:
            return self._count_with_regex(text)

        doc    = _nlp(text.lower())
        counts: Dict[str, int] = {}
        for token in doc:
            key = None
            if token.lemma_ in self.fillers:
                key = token.lemma_
            elif token.text in self.fillers:
                key = token.text
            if key:
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_with_regex(self, text: str) -> Dict[str, int]:
        text_lower = text.lower()
        counts: Dict[str, int] = {}
        for filler, pattern in self._patterns.items():
            n = len(pattern.findall(text_lower))
            if n:
                counts[filler] = n
        return counts

    # Weighted total 

    def _weighted_total(self, counts: Dict[str, int]) -> float:
        total = 0.0
        for filler, n in counts.items():
            weight = self.high_weight_mult if filler in self.high_weight else 1.0
            total += n * weight
        return total

    # Main compute 

    def compute(self, transcript_text: str, use_spacy: bool = True) -> Dict:
        """
        Parameters
        ----------
        transcript_text : raw transcript string (no timestamps needed)
        use_spacy       : prefer spaCy lemmatisation when available

        Returns
        -------
        {total_words, total_fillers, weighted_fillers, filler_rate,
         weighted_rate, breakdown, score, method}
        """
        if not isinstance(transcript_text, str):
            raise ValueError("transcript_text must be a string")

        total_words = len(transcript_text.split())

        filler_counts = (
            self._count_with_spacy(transcript_text)
            if use_spacy
            else self._count_with_regex(transcript_text)
        )
        raw_total      = sum(filler_counts.values())
        weighted_total = self._weighted_total(filler_counts)

        filler_rate   = raw_total      / total_words if total_words > 0 else 0.0
        weighted_rate = weighted_total / total_words if total_words > 0 else 0.0

        score = round(min(weighted_rate / self.max_filler_rate, 1.0), 3)

        result = {
            "total_words":     total_words,
            "total_fillers":   raw_total,
            "weighted_fillers": round(weighted_total, 2),
            "filler_rate":     round(filler_rate,   4),
            "weighted_rate":   round(weighted_rate,  4),
            "breakdown":       filler_counts,
            "score":           score,            # 0 = clean, 1 = filler-heavy
            "method":          "spacy" if (use_spacy and SPACY_AVAILABLE) else "regex",
        }
        self.last_result = result
        return result

    # Persistence 

    def save(self, path: str) -> None:
        data = {
            "fillers":          list(self.fillers),
            "high_weight":      list(self.high_weight),
            "max_filler_rate":  self.max_filler_rate,
            "high_weight_mult": self.high_weight_mult,
            "last_result":      self.last_result,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.fillers          = set(data["fillers"])
        self.high_weight      = set(data.get("high_weight", list(HIGH_WEIGHT_FILLERS)))
        self.max_filler_rate  = data["max_filler_rate"]
        self.high_weight_mult = data.get("high_weight_mult", 2.0)
        self._patterns        = self._compile(self.fillers)
        self.last_result      = data.get("last_result")


# Standalone runner 

def main():
    from pathlib import Path

    DATA_DIR   = Path("baby-data/transcripts")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "filler_metrics.json"
    LABELS     = ["low", "medium", "high"]

    agent = FillerPatternsAgent(max_filler_rate=0.12)

    print("=== Filler Word Analysis ===")
    print(f"  Method: {'spaCy' if SPACY_AVAILABLE else 'regex (spaCy not installed)'}\n")

    for label in LABELS:
        data            = json.load(open(DATA_DIR / f"{label}_transcript.json"))
        transcript_text = data["text"]

        result = agent.compute(transcript_text)
        print(f"  [{label.upper()}]")
        print(f"    total_words      : {result['total_words']}")
        print(f"    total_fillers    : {result['total_fillers']}")
        print(f"    weighted_fillers : {result['weighted_fillers']}")
        print(f"    filler_rate      : {result['filler_rate']:.2%}  (raw)")
        print(f"    weighted_rate    : {result['weighted_rate']:.2%}  (used for score)")
        print(f"    score            : {result['score']}  (0 = clean, 1 = filler-heavy)")
        print(f"    breakdown        : {result['breakdown']}")
        print()

    agent.save(SAVE_PATH)
    print(f"Saved config + last result → {SAVE_PATH}")

    agent2 = FillerPatternsAgent()
    agent2.load(SAVE_PATH)
    print(f"Re-loaded fillers     : {sorted(agent2.fillers)}")
    print(f"Re-loaded last_result : {agent2.last_result}")


if __name__ == "__main__":
    main()