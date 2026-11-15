"""Build speech, caption and cue files for the RAGTIME approach film."""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import soundfile
from kokoro_onnx import Kokoro

SAMPLE_RATE = 24_000
VOICE = "af_heart"


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    minutes, remaining = divmod(milliseconds, 60_000)
    whole_seconds, fraction = divmod(remaining, 1000)
    return f"00:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="Directory containing the ONNX model and voices")
    args = parser.parse_args()
    build = args.build_dir
    script = json.loads(Path(__file__).with_name("story.json").read_text(encoding="utf-8"))
    synthesizer = Kokoro(
        str(build / "kokoro-v1.0.int8.onnx"),
        str(build / "voices-v1.0.bin"),
    )
    parts = []
    captions = ["WEBVTT\n"]
    timeline = {}
    sample_count = 0
    for index, cue in enumerate(script):
        gap = cue.get("gap", 0.20 if index else 0.55)
        silence = np.zeros(round(gap * SAMPLE_RATE), dtype=np.float32)
        parts.append(silence)
        sample_count += len(silence)
        start = sample_count / SAMPLE_RATE

        speech, sample_rate = synthesizer.create(
            cue.get("spoken", cue["text"]),
            voice=VOICE,
            speed=1.06,
            lang="en-us",
        )
        if sample_rate != SAMPLE_RATE or not len(speech):
            raise ValueError(f"Unexpected audio for {cue['id']}")
        parts.append(speech)
        sample_count += len(speech)
        end = sample_count / SAMPLE_RATE
        timeline[cue["id"]] = {"start": start, "end": end}
        caption = textwrap.fill(cue["text"], width=52, break_long_words=False)
        captions.append(f"{timestamp(start)} --> {timestamp(end)}\n{caption}\n")
        print(f"{start:5.1f}–{end:5.1f}  {cue['id']}", flush=True)

    tail_samples = round(1.8 * SAMPLE_RATE)
    parts.append(np.zeros(tail_samples, dtype=np.float32))
    sample_count += tail_samples
    audio = np.concatenate(parts)
    soundfile.write(build / "narration.wav", audio, SAMPLE_RATE)
    (build / "captions.vtt").write_text("\n".join(captions), encoding="utf-8")
    result = {"duration": sample_count / SAMPLE_RATE, "cues": timeline}
    (build / "timeline.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
