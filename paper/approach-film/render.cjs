const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { once } = require("node:events");
const { pathToFileURL } = require("node:url");

const FPS = 24;
const WIDTH = 1600;
const HEIGHT = 900;

async function render(build, previewOnly) {
  const playwrightPath = process.env.RAGTIME_FILM_PLAYWRIGHT
    ?? require.resolve("playwright", { paths: [build] });
  const { chromium } = require(playwrightPath);
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.RAGTIME_FILM_CHROME,
  });
  let encoder;
  try {
    const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const timeline = JSON.parse(fs.readFileSync(path.join(build, "timeline.json"), "utf8"));
    await page.goto(pathToFileURL(path.join(__dirname, "film.html")).href);
    await page.evaluate(async (timeline) => {
      await document.fonts.ready;
      await Promise.all(Array.from(document.images, (image) => image.decode()));
      window.setFilmTimeline(timeline);
    }, timeline);
    const session = await page.context().newCDPSession(page);

    async function frame(time, quality) {
      await page.evaluate((time) => window.drawFilm(time), time);
      if (errors.length) throw new Error(errors.join("\n"));
      const { data } = await session.send("Page.captureScreenshot", { format: "jpeg", quality });
      return Buffer.from(data, "base64");
    }

    if (previewOnly) {
      const samples = {
        opening: timeline.cues.request.end - 0.2,
        queries: timeline.cues.search.end - 0.2,
        selection: timeline.cues.select.end - 0.2,
        writing: timeline.cues.finding.end - 0.2,
        coverage: timeline.cues.same.end - 0.2,
        results: timeline.cues.end.end - 0.2,
      };
      for (const [name, time] of Object.entries(samples)) {
        fs.writeFileSync(path.join(build, name + ".jpg"), await frame(time, 94));
      }
      return;
    }

    const out = path.join(build, "approach.mp4");
    encoder = spawn("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "pipe:0",
      "-i", path.join(build, "narration.wav"),
      "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "20",
      "-pix_fmt", "yuv420p",
      "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
      "-af", "loudnorm=I=-17:TP=-2:LRA=8",
      "-movflags", "+faststart", "-shortest", out,
    ], { stdio: ["pipe", "inherit", "inherit"] });
    const completed = once(encoder, "close");
    // An encoder failure also closes the image pipe. Report the encoder result below.
    encoder.stdin.on("error", (error) => {
      if (error.code !== "EPIPE") console.error(error);
    });

    const frames = Math.ceil(timeline.duration * FPS);
    for (let index = 0; index < frames; index++) {
      if (encoder.exitCode !== null || encoder.stdin.destroyed) {
        throw new Error("FFmpeg stopped before all frames were written");
      }
      const image = await frame(index / FPS, 93);
      if (!encoder.stdin.write(image)) await once(encoder.stdin, "drain");
      if (index > 0 && index % (FPS * 10) === 0) console.log(`Rendered ${index / FPS}s`);
    }
    encoder.stdin.end();
    const [code, signal] = await completed;
    if (code !== 0) throw new Error(`FFmpeg ended with ${signal ?? code}`);
    console.log(out);
  } finally {
    if (encoder?.exitCode === null) encoder.kill();
    await browser.close();
  }
}

const [build, mode] = process.argv.slice(2);
if (!build || (mode && mode !== "--preview")) {
  console.error("Usage: node render.cjs BUILD_DIR [--preview]");
  process.exit(2);
}
render(path.resolve(build), mode === "--preview").catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
