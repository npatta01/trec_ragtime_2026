# TREC RAGTIME 2026 approach video

The workshop example follows Topic 2014, African penguin conservation, from an English query to a cited finding and preliminary AutoARGUE judgments. The topic trace follows the [workshop slides](../../slides/ragtime2026.qmd). The [notebook paper](../ragtime2026-np-draft.pdf) describes the system and the [organizer results](../../results/2026-11-02/README.md) supply the aggregate scores.

- [Narration and captions](story.json)
- [Layout and animation](film.html)
- [African penguin artwork](../../slides/art/african-penguin.jpg)
- [Submitted report for Topic 2014](../../submissions/np-nugget-sol-v1/NP.np-nugget-sol-v1.ragtime26.jsonl#L15)

## Render the video

Prepare a build directory outside the checkout and place the public [Kokoro ONNX model](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx) and [voice bank](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) in it. Narration uses the same local synthetic `af_heart` voice as the TREC UserSim video. Use Python with `kokoro-onnx==0.6.1` and `soundfile==0.13.1`, Node with Playwright and Chromium, and FFmpeg with libx264. Install Playwright in the build directory or set `RAGTIME_FILM_PLAYWRIGHT` to its module path. Set `RAGTIME_FILM_CHROME` to use an installed Chrome binary.

From the repository root, set `film_build_dir` to the build directory and run the following.

```bash
sops decrypt --output .env config/secrets.sops.env
python -B paper/approach-film/narrate.py "$film_build_dir"
node paper/approach-film/render.cjs "$film_build_dir" --preview
node paper/approach-film/render.cjs "$film_build_dir"
```

The scripts render locally. Update the site assets after verifying the new export.

```bash
install -m 644 "$film_build_dir/approach.mp4" assets/ragtime-approach.mp4
install -m 644 "$film_build_dir/opening.jpg" assets/ragtime-approach.jpg
install -m 644 "$film_build_dir/captions.vtt" assets/ragtime-approach.vtt
```

Keep the video duration and transcript on the project page in sync with the export.
