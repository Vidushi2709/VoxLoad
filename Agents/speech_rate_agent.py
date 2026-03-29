"""
SpeechRateAgent
---------------
Input  : word timestamps  (List[Dict] with 'word', 'start', 'end')
Type   : Algorithmic
Library: numpy  (librosa optional — used only for extra audio features)

Score semantics
---------------
  0.0 → very fast, steady speech  (low cognitive load)
  1.0 → very slow, highly variable speech  (high cognitive load)

Two components:
  speed_score    – how slow the speaker is relative to calibration range
  variance_score – variability of WPM across 20-second windows

TEMPORARY CALIBRATION
---------------------
Uses hardcoded WPM range (80–200) based on linguistic literature until
30+ recordings are available for proper fitting.  This prevents overfitting
to the limited initial dataset.

Literature baseline: normal conversational speech is 120–180 WPM
"""

import json
import numpy as np
from typing import List, Dict, Optional

# Import z-score helper
from z_score import compute_z_score

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class SpeechRateAgent:
    def __init__(self):
        # TEMPORARY CALIBRATION: hardcoded ranges based on linguistic literature
        # Normal conversational speech: 120–180 WPM
        # TODO: Replace with fit() once dataset size is adequate
        self.min_wpm: Optional[float] = 80.0    # slow/struggling speech
        self.max_wpm: Optional[float] = 200.0   # fast/fluent speech
        self.fitted:  bool            = True    # mark as fitted to skip fit() requirement

        # Sliding-window config
        self._window_sec: int = 20
        self._step_sec:   int = 5

        # Variance normaliser: variance (in WPM²) at which variance_score = 1.0
        self._max_variance: float = 400.0

        # Rush threshold: WPM above this value triggers the rush penalty
        # (only matters when variance is also high)
        self._rush_wpm_threshold: float = 110.0

    # Calibration 

    def fit(self, all_words_list: List[List[Dict]]) -> None:
        """
        Compute the WPM range from a representative dataset.
        Call once before run(); or use load() to restore a saved calibration.
        
        ⚠️  WARNING: Requires at least 30 recordings for reliable calibration.
        With fewer recordings, will overfit to your limited dataset.
        Use hardcoded defaults (80–200 WPM) until you have adequate data.
        """
        if len(all_words_list) < 30:
            print(
                f"⚠️  WARNING: fit() called with only {len(all_words_list)} recordings. "
                f"Recommend 30+ for reliable calibration. Using hardcoded defaults: "
                f"80–200 WPM."
            )
            return
        
        wpm_values = []
        for words in all_words_list:
            if len(words) < 2:
                continue
            duration_min = (words[-1]["end"] - words[0]["start"]) / 60.0
            if duration_min > 0:
                wpm_values.append(len(words) / duration_min)

        if not wpm_values:
            raise ValueError("No valid word lists supplied to fit()")

        self.min_wpm = min(wpm_values)
        self.max_wpm = max(wpm_values)
        self.fitted  = True
        print(f"✓ Fitted on {len(wpm_values)} recordings — WPM range: {self.min_wpm:.1f} → {self.max_wpm:.1f}")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "min_wpm":            self.min_wpm,
                    "max_wpm":            self.max_wpm,
                    "rush_wpm_threshold": self._rush_wpm_threshold,
                },
                f, indent=2,
            )

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.min_wpm              = data["min_wpm"]
        self.max_wpm              = data["max_wpm"]
        self._rush_wpm_threshold  = data.get("rush_wpm_threshold", 110.0)
        self.fitted               = True

    # Optional audio features (requires librosa) 

    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Pull speaking-rate proxies directly from the audio file.
        Returns an empty dict if librosa is not installed.
        """
        if not LIBROSA_AVAILABLE:
            print("librosa not installed — skipping audio feature extraction.")
            return {}

        y, sr = librosa.load(audio_path, sr=None)

        # voiced-frame ratio as a rough speech-activity proxy
        rms          = librosa.feature.rms(y=y)[0]
        voiced_ratio = float(np.mean(rms > rms.mean()))

        # tempo from onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _  = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        return {
            "duration_sec":        round(len(y) / sr, 2),
            "voiced_ratio":        round(voiced_ratio, 3),
            "estimated_tempo_bpm": round(float(tempo), 1),
        }

    # Main compute 

    def run(self, words: List[Dict], audio_path: Optional[str] = None, speaker_id: Optional[str] = None, baselines: Optional[Dict] = None) -> Dict:
        """
        Parameters
        ----------
        words      : list of {word, start, end} dicts from Whisper
        audio_path : optional path for librosa feature extraction
        speaker_id : (optional) speaker identifier for z-score computation
        baselines  : (optional) {speaker_id: {agent: baseline, ...}, "_population_std": {agent: std, ...}}

        Returns
        -------
        {wpm, wpm_variance, wpm_windows, slow_score, variance_score, rush_score, raw_score, score (z-score), [audio_features]}
        
        Note: Currently uses hardcoded WPM calibration (80–200) for consistency.
        Will be replaced with fitted calibration once 30+ recordings are available.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() or load() before run() — or use hardcoded defaults")
        if not words:
            return {
                "wpm": 0.0, "wpm_variance": 0.0,
                "wpm_windows": [], "raw_score": 0.5, "score": 0.0,
            }

        # Guard: reject recordings that are too short for reliable WPM estimation
        MIN_DURATION_SEC = 20.0
        MIN_WORDS = 25
        
        duration_sec = words[-1]["end"] - words[0]["start"] if len(words) >= 2 else 0
        if duration_sec < MIN_DURATION_SEC or len(words) < MIN_WORDS:
            return {
                "wpm": round(len(words) / (duration_sec / 60.0) if duration_sec > 0 else 0.0, 1),
                "wpm_variance": 0.0,
                "wpm_windows": [],
                "slow_score": 0.0,
                "variance_score": 0.0,
                "rush_score": 0.0,
                "raw_score": 0.5,
                "score": 0.0,
                "unreliable": True,
                "unreliable_reason": f"Recording too short ({duration_sec:.1f}s, {len(words)} words)"
            }

        duration_min = duration_sec / 60.0
        avg_wpm      = len(words) / duration_min if duration_min > 0 else 0.0

        # Sliding-window WPM variance 
        window, step = self._window_sec, self._step_sec
        wpm_windows  = []
        t            = words[0]["start"]
        end_time     = words[-1]["end"]
        while t + window <= end_time:
            count = sum(1 for w in words if t <= w["start"] < t + window)
            wpm_windows.append(count * (60.0 / window))
            t += step

        variance = float(np.var(wpm_windows)) if wpm_windows else 0.0

        # Score components
        wpm_range = (self.max_wpm - self.min_wpm) if self.max_wpm != self.min_wpm else 1.0

        # 1. slow_score: 0 = fast (low load), 1 = slow (high load)
        slow_score = 1.0 - max(0.0, min(1.0, (avg_wpm - self.min_wpm) / wpm_range))

        # 2. variance_score: 0 = steady, 1 = highly variable
        variance_score = min(variance / self._max_variance, 1.0)

        # 3. rush_score: fast + erratic → elevated load
        rush_factor = max(0.0, avg_wpm - self._rush_wpm_threshold) / self._rush_wpm_threshold
        rush_factor = min(1.0, rush_factor)
        rush_score  = rush_factor * variance_score   # zero if speech is steady

        # Weighted blend
        raw_score = round(slow_score * 0.45 + variance_score * 0.35 + rush_score * 0.20, 3)

        # Compute z-score if baselines available
        z_score = raw_score
        if speaker_id and baselines:
            speaker_baseline = baselines.get(speaker_id, {})
            baseline = speaker_baseline.get("speech_rate", raw_score)
            z_score = compute_z_score(raw_score, baseline, "speech_rate")

        result: Dict = {
            "wpm":            round(avg_wpm,  1),
            "wpm_variance":   round(variance, 2),
            "wpm_windows":    [round(w, 1) for w in wpm_windows],
            "slow_score":     round(slow_score,     3),
            "variance_score": round(variance_score, 3),
            "rush_score":     round(rush_score,     3),
            "raw_score":      raw_score,
            "score":          z_score,           # z-score
        }

        if audio_path:
            result["audio_features"] = self.extract_audio_features(audio_path)

        return result


# Standalone runner 

def main():
    import os
    from pathlib import Path

    DATA_DIR   = Path("baby-data/transcripts")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "speech_rate_agent.json"
    LABELS     = ["low", "medium", "high"]

    # 1. Load all transcripts
    all_words: dict = {}
    for label in LABELS:
        data             = json.load(open(DATA_DIR / f"{label}_transcript.json"))
        all_words[label] = data["words"]

    # 2. Fit once on the full dataset
    agent = SpeechRateAgent()
    agent.fit(list(all_words.values()))
    agent.save(SAVE_PATH)
    print(f"Calibration saved → {SAVE_PATH}\n")

    # 3. Load & score each transcript
    agent2 = SpeechRateAgent()
    agent2.load(SAVE_PATH)

    print("=== Speech Rate Analysis ===\n")
    for label in LABELS:
        audio_wav = DATA_DIR / f"{label}.wav"
        audio_mp4 = DATA_DIR / f"{label}.mp4"
        audio_path = (
            str(audio_wav) if audio_wav.exists()
            else str(audio_mp4) if audio_mp4.exists()
            else None
        )

        r = agent2.run(all_words[label], audio_path=audio_path)
        print(f"  [{label.upper()}]")
        print(f"    WPM      : {r['wpm']}")
        print(f"    variance : {r['wpm_variance']}")
        print(f"    score    : {r['score']}  (0 = fast/low-load, 1 = slow/high-load)")
        if "audio_features" in r:
            print(f"    audio    : {r['audio_features']}")
        print()


if __name__ == "__main__":
    main()