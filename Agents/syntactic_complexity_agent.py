"""
SyntacticComplexityAgent
------------------------
Input  : transcript text only  (plain string)
Type   : Algorithmic  (spaCy dependency parse — no LLM calls)
Library: spaCy en_core_web_sm

Score semantics
---------------
  0.0 → highly complex syntax  (expert speaker, low cognitive load)
  1.0 → very simple syntax     (struggling / overloaded speaker, high load)

Why syntactic complexity matters
---------------------------------
A fluent expert produces long sentences, deep dependency trees, and rich
subordination even when speaking quickly.  A cognitively overloaded speaker
fragments their output into short, simple clauses — even if they are not
pausing heavily.  This dimension is not captured by any of the other agents.

Three sub-measures
------------------
  1. avg_sentence_length  — mean words per sentence
     Long sentences → more ideas held in working memory → lower load signal
     (counter-intuitive: expert = longer sentences)

  2. avg_tree_depth       — mean dependency-parse tree depth per sentence
     Deeper trees → more nested structure → expert language use

  3. subordination_ratio  — fraction of tokens with a subordinating
     dependency relation (advcl, relcl, ccomp, xcomp, acl, mark)
     More subordination → richer clause embedding → lower load signal

All three are inverted so that HIGH complexity → LOW score (low load).

Calibration defaults
--------------------
The normalisation ceilings are tuned to typical conversational English.
Override via the constructor for domain-specific calibration.
"""

import re
import json
from typing import List, Dict, Optional


try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# Dependency relations that signal embedded / subordinate clauses
_SUBORDINATING_DEPS = {
    "advcl",   # adverbial clause modifier
    "relcl",   # relative clause modifier
    "ccomp",   # clausal complement
    "xcomp",   # open clausal complement
    "acl",     # adjectival clause
    "mark",    # subordinating conjunction
}


def _tree_depth(token) -> int:
    """Recursively compute the depth of a dependency sub-tree."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_tree_depth(c) for c in children)


class SyntacticComplexityAgent:
    """
    Pure-algorithmic agent: uses spaCy's dependency parser only.
    Falls back to a length-only heuristic if spaCy is unavailable.
    """

    def __init__(
        self,
        # Calibration ceilings (values at or above these → maximum contribution)
        max_avg_sentence_len:   float = 25.0,   # words per sentence
        max_avg_tree_depth:     float = 8.0,    # dependency depth
        max_subordination_rate: float = 0.20,   # fraction of tokens
        # Sub-measure weights (must sum to 1.0)
        w_sentence_len:   float = 0.40,
        w_tree_depth:     float = 0.35,
        w_subordination:  float = 0.25,
        # Minimum sentences to trust the estimate
        min_sentences: int = 3,
    ):
        self.max_avg_sentence_len   = max_avg_sentence_len
        self.max_avg_tree_depth     = max_avg_tree_depth
        self.max_subordination_rate = max_subordination_rate
        self.w_sentence_len         = w_sentence_len
        self.w_tree_depth           = w_tree_depth
        self.w_subordination        = w_subordination
        self.min_sentences          = min_sentences
        self.last_result: Optional[Dict] = None

    # Core computation 

    def _analyse_with_spacy(self, text: str) -> Dict:
        """Full dependency-parse analysis."""
        doc = _nlp(text)
        sents = list(doc.sents)
        if not sents:
            return self._empty_result("no sentences found")

        sent_lengths  = [len(sent) for sent in sents]
        tree_depths   = [_tree_depth(sent.root) for sent in sents]
        sub_tokens    = sum(
            1 for tok in doc if tok.dep_ in _SUBORDINATING_DEPS
        )

        n_sents = len(sents)
        n_tokens = len(doc)

        avg_len   = sum(sent_lengths) / n_sents
        avg_depth = sum(tree_depths)  / n_sents
        sub_rate  = sub_tokens / n_tokens if n_tokens > 0 else 0.0

        return {
            "avg_sentence_len":    round(avg_len,   2),
            "avg_tree_depth":      round(avg_depth, 3),
            "subordination_rate":  round(sub_rate,  4),
            "sentence_count":      n_sents,
            "token_count":         n_tokens,
            "method":              "spacy",
        }

    def _analyse_fallback(self, text: str) -> Dict:
        """
        Lightweight fallback when spaCy is not installed.
        Uses sentence splitting by punctuation and average word length
        as a very rough tree-depth proxy.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return self._empty_result("no sentences found (fallback)")

        words_per_sent = [len(s.split()) for s in sentences]
        avg_len = sum(words_per_sent) / len(sentences)

        # Rough subordination proxy: count "that/which/who/because/although/…"
        sub_words = re.findall(
            r"\b(that|which|who|whom|whose|because|although|though|if|"
            r"since|when|while|where|unless|until|so that)\b",
            text.lower(),
        )
        total_words = len(text.split())
        sub_rate = len(sub_words) / total_words if total_words > 0 else 0.0

        # Depth proxy: avg comma count per sentence (more clauses ≈ more commas)
        avg_depth_proxy = sum(s.count(",") for s in sentences) / len(sentences)

        return {
            "avg_sentence_len":    round(avg_len, 2),
            "avg_tree_depth":      round(avg_depth_proxy, 3),  # proxy, not real depth
            "subordination_rate":  round(sub_rate, 4),
            "sentence_count":      len(sentences),
            "token_count":         total_words,
            "method":              "fallback_regex",
        }

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        return {
            "avg_sentence_len":   0.0,
            "avg_tree_depth":     0.0,
            "subordination_rate": 0.0,
            "sentence_count":     0,
            "token_count":        0,
            "method":             f"empty ({reason})",
        }

    def compute(self, transcript_text: str) -> Dict:
        """
        Parameters
        ----------
        transcript_text : raw transcript string

        Returns
        -------
        {
          avg_sentence_len,  avg_tree_depth,  subordination_rate,
          sentence_count,    token_count,     method,
          complexity_score,  score
        }

        ``score``
            0.0 = highly complex speech  (expert / low cognitive load)
            1.0 = very simple speech     (overloaded / high cognitive load)
        """
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            result = self._empty_result("empty input")
            result["complexity_score"] = 0.0
            result["score"]            = 0.5   # agnostic when no data
            self.last_result = result
            return result

        metrics = (
            self._analyse_with_spacy(transcript_text)
            if SPACY_AVAILABLE
            else self._analyse_fallback(transcript_text)
        )

        # Warn if too few sentences to trust the result
        if metrics["sentence_count"] < self.min_sentences:
            print(
                f"  [syntactic] WARNING: only {metrics['sentence_count']} sentence(s) "
                f"found — complexity score may be unreliable."
            )

        # Normalised sub-scores: 1.0 = maximum complexity (expert)
        len_score  = min(metrics["avg_sentence_len"]   / self.max_avg_sentence_len,   1.0)
        dep_score  = min(metrics["avg_tree_depth"]      / self.max_avg_tree_depth,     1.0)
        sub_score  = min(metrics["subordination_rate"]  / self.max_subordination_rate, 1.0)

        # Weighted composite complexity (0 = simple, 1 = complex)
        complexity_score = round(
            self.w_sentence_len  * len_score
            + self.w_tree_depth  * dep_score
            + self.w_subordination * sub_score,
            3,
        )

        # Invert: simple speech = HIGH cognitive load score
        score = round(1.0 - complexity_score, 3)

        result = {
            **metrics,
            "complexity_score": complexity_score,   # 0 = simple, 1 = complex
            "score":            score,               # 0 = complex (expert), 1 = simple (overloaded)
        }
        self.last_result = result
        return result

    # Persistence 

    def save(self, path: str) -> None:
        data = {
            "max_avg_sentence_len":   self.max_avg_sentence_len,
            "max_avg_tree_depth":     self.max_avg_tree_depth,
            "max_subordination_rate": self.max_subordination_rate,
            "w_sentence_len":         self.w_sentence_len,
            "w_tree_depth":           self.w_tree_depth,
            "w_subordination":        self.w_subordination,
            "last_result":            self.last_result,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.max_avg_sentence_len   = data["max_avg_sentence_len"]
        self.max_avg_tree_depth     = data["max_avg_tree_depth"]
        self.max_subordination_rate = data["max_subordination_rate"]
        self.w_sentence_len         = data["w_sentence_len"]
        self.w_tree_depth           = data["w_tree_depth"]
        self.w_subordination        = data["w_subordination"]
        self.last_result            = data.get("last_result")


# Standalone runner

def main():
    from pathlib import Path

    DATA_DIR   = Path("baby-data/transcripts")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "syntactic_complexity_metrics.json"
    LABELS     = ["low", "medium", "high"]

    if not SPACY_AVAILABLE:
        print("WARNING: spaCy not available — using fallback regex analysis.")

    agent = SyntacticComplexityAgent()

    print("=== Syntactic Complexity Analysis ===\n")
    print(f"  Method: {'spaCy dependency parse' if SPACY_AVAILABLE else 'fallback regex'}\n")

    for label in LABELS:
        path = DATA_DIR / f"{label}_transcript.json"
        if not path.exists():
            print(f"  [{label.upper()}] transcript not found, skipping.")
            continue
        data = json.load(open(path))
        text = data["text"]

        result = agent.compute(text)
        print(f"  [{label.upper()}]")
        print(f"    sentence_count      : {result['sentence_count']}")
        print(f"    avg_sentence_len    : {result['avg_sentence_len']} words")
        print(f"    avg_tree_depth      : {result['avg_tree_depth']}")
        print(f"    subordination_rate  : {result['subordination_rate']:.2%}")
        print(f"    complexity_score    : {result['complexity_score']}  (0=simple, 1=complex)")
        print(f"    score               : {result['score']}  (0=expert/low-load, 1=overloaded/high-load)")
        print()

    agent.save(SAVE_PATH)
    print(f"Saved config + last result → {SAVE_PATH}")


if __name__ == "__main__":
    main()
