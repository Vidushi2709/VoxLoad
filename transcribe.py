from faster_whisper import WhisperModel
import json
from pathlib import Path

# GPU with int8 quantization — faster and lighter
print("Loading model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

for label in ["low", "medium", "high"]:
    print(f"Transcribing {label}...")

    segments, info = model.transcribe(
        str(Path("baby-data") / f"{label}.wav"),
        word_timestamps=True,
        language="en"
    )

    output = {
        "label": label,
        "text": "",
        "words": []
    }

    full_text = []
    for segment in segments:
        full_text.append(segment.text)
        for word in segment.words:
            output["words"].append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3)
            })

    output["text"] = " ".join(full_text)

    out_path = Path("baby-data") / f"{label}_transcript.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ {len(output['words'])} words found")
    print(f"  Preview: {output['text'][:120]}")
    print()

print("All done — check baby-data/ for the 3 JSON files")