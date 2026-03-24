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
"""

import json
import numpy as np
from typing import List, Dict, Optional


try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class SpeechRateAgent:
    def __init__(self):
        self.min_wpm: Optional[float] = None
        self.max_wpm: Optional[float] = None
        self.fitted:  bool            = False

        # Sliding-window config
        self._window_sec: int = 20
        self._step_sec:   int = 5

        # Variance normaliser: variance (in WPM²) at which variance_score = 1.0
        # Empirically ~400 WPM² covers most natural variation.
        self._max_variance: float = 400.0

    # Calibration 

    def fit(self, all_words_list: List[List[Dict]]) -> None:
        """
        Compute the WPM range from a representative dataset.
        Call once before run(); or use load() to restore a saved calibration.
        """
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
        print(f"Fitted — WPM range: {self.min_wpm:.1f} → {self.max_wpm:.1f}")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {"min_wpm": self.min_wpm, "max_wpm": self.max_wpm},
                f, indent=2,
            )

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.min_wpm = data["min_wpm"]
        self.max_wpm = data["max_wpm"]
        self.fitted  = True

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

    def run(self, words: List[Dict], audio_path: Optional[str] = None) -> Dict:
        """
        Parameters
        ----------
        words      : list of {word, start, end} dicts from Whisper
        audio_path : optional path for librosa feature extraction

        Returns
        -------
        {wpm, wpm_variance, wpm_windows, score, [audio_features]}
        """
        if not self.fitted:
            raise RuntimeError("Call fit() or load() before run()")
        if not words:
            return {
                "wpm": 0.0, "wpm_variance": 0.0,
                "wpm_windows": [], "score": 0.5,
            }

        duration_min = (words[-1]["end"] - words[0]["start"]) / 60.0
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

        # speed_score: 0 = fast (low load), 1 = slow (high load)
        speed_score = (avg_wpm - self.min_wpm) / wpm_range
        speed_score = 1.0 - max(0.0, min(1.0, speed_score))  # invert: slower = higher load

        # variance_score: 0 = steady, 1 = highly variable
        variance_score = min(variance / self._max_variance, 1.0)

        score = round(speed_score * 0.6 + variance_score * 0.4, 3)

        result: Dict = {
            "wpm":          round(avg_wpm,  1),
            "wpm_variance": round(variance, 2),
            "wpm_windows":  [round(w, 1) for w in wpm_windows],
            "score":        score,
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