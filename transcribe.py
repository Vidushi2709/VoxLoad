from faster_whisper import WhisperModel
import json
from pathlib import Path

print("Loading model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

for label in ["low", "medium", "high"]:
    print(f"Transcribing {label}...")

    segments, info = model.transcribe(
        str(Path("baby-data/raw_audio_to_wav_format") / f"{label}.wav"),
        word_timestamps=True,
        language="en",
        beam_size=5,
        vad_filter=True,               # strips silence before alignment
        vad_parameters=dict(
            min_silence_duration_ms=100,   # detect short pauses
            speech_pad_ms=100,
        ),
        condition_on_previous_text=False,  # prevents hallucination loops
        temperature=0.0,               # greedy decode = more stable timestamps
    )

    output = {"label": label, "text": "", "words": []}
    full_text = []

    for segment in segments:
        full_text.append(segment.text.strip())
        for word in segment.words:
            output["words"].append({
                "word":  word.word.strip(),
                "start": round(word.start, 3),
                "end":   round(word.end,   3)
            })

    output["text"] = " ".join(full_text)

    # quick quality check inline
    words = output["words"]
    gaps = [(words[i]["start"] - words[i-1]["end"]) * 1000
            for i in range(1, len(words))]
    nonzero = sum(1 for g in gaps if g > 0)
    print(f"  ✓ {len(words)} words  |  nonzero gaps: {nonzero}/{len(gaps)}")

    out_path = Path("baby-data/transcripts") / f"{label}_transcript.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

print("\nDone")