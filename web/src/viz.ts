/**
 * Canvas visualizations: the audio level strip and the speaker timeline.
 *
 * These are the two views that make the pipeline's behaviour legible rather
 * than merely reported. The level strip shows *endpointing deciding* — where
 * speech began, where the silence window closed an utterance — which is
 * otherwise a number in a config file. The speaker timeline shows diarization
 * as the thing it actually is: bands of time attributed to people.
 *
 * Both draw on `requestAnimationFrame` from a ring buffer and never touch the
 * store. Level samples arrive 20 times a second; pushing those through
 * application state would spend main thread on reconciliation that competes
 * with `AudioWorklet` message delivery, and dropped input audio cannot be
 * recovered.
 */

import { speakerColorIndex, type Utterance } from "./state";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Size a canvas for the device pixel ratio.
 *
 * Without this a canvas on a retina display renders at half resolution and
 * looks soft next to the text beside it — which reads as sloppiness even to
 * someone who could not say why.
 */
function fit(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

interface Sample {
  level: number;
  speech: boolean;
}

/** Scrolling input level, coloured by whether the endpointer heard speech. */
export class LevelStrip {
  private readonly samples: Sample[] = [];
  private readonly capacity: number;
  private raf = 0;
  private speaking = false;
  private running = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    capacity = 320,
  ) {
    this.capacity = capacity;
  }

  setSpeaking(speaking: boolean): void {
    this.speaking = speaking;
  }

  push(level: number): void {
    this.samples.push({ level: Math.min(1, level), speech: this.speaking });
    if (this.samples.length > this.capacity) this.samples.shift();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    const draw = () => {
      this.render();
      this.raf = requestAnimationFrame(draw);
    };
    this.raf = requestAnimationFrame(draw);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.render();
  }

  clear(): void {
    this.samples.length = 0;
    this.render();
  }

  render(): void {
    const ctx = fit(this.canvas);
    if (!ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    const { width, height } = rect;
    const mid = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Baseline, so an empty strip still reads as "a signal view, currently quiet"
    // rather than as a broken element.
    ctx.strokeStyle = cssVar("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    if (this.samples.length === 0) return;

    const speechColor = cssVar("--ok");
    const quietColor = cssVar("--border-strong");
    const step = width / this.capacity;
    const barWidth = Math.max(1, step - 1);

    this.samples.forEach((sample, index) => {
      const x = index * step;
      // Perceptual, not linear: a linear level meter spends most of its range
      // on amplitudes nobody can hear and looks dead during normal speech.
      const amplitude = Math.pow(sample.level, 0.6) * (mid - 3);
      ctx.fillStyle = sample.speech ? speechColor : quietColor;
      ctx.globalAlpha = sample.speech ? 0.9 : 0.5;
      ctx.fillRect(x, mid - amplitude, barWidth, amplitude * 2);
    });
    ctx.globalAlpha = 1;
  }
}

/** Who held the floor, when. */
export class SpeakerTimeline {
  constructor(private readonly canvas: HTMLCanvasElement) {}

  render(utterances: readonly Utterance[], totalSeconds: number): void {
    const ctx = fit(this.canvas);
    if (!ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    const { width, height } = rect;
    ctx.clearRect(0, 0, width, height);

    if (utterances.length === 0 || totalSeconds <= 0) {
      ctx.fillStyle = cssVar("--surface-3");
      ctx.fillRect(0, height / 2 - 3, width, 6);
      return;
    }

    const speakers = [...new Set(utterances.map((u) => u.speaker || "?"))];
    const laneHeight = Math.min(14, Math.max(6, (height - 8) / speakers.length));
    const gap = 2;

    speakers.forEach((speaker, lane) => {
      const y = 4 + lane * (laneHeight + gap);
      const color = cssVar(`--spk-${speakerColorIndex(speaker)}`);

      // Lane background, so a speaker who says one word still has a visible row
      ctx.fillStyle = cssVar("--surface-3");
      ctx.globalAlpha = 0.45;
      ctx.fillRect(0, y, width, laneHeight);
      ctx.globalAlpha = 1;

      ctx.fillStyle = color;
      for (const utterance of utterances) {
        if ((utterance.speaker || "?") !== speaker) continue;
        const x = (utterance.start / totalSeconds) * width;
        const w = Math.max(2, ((utterance.end - utterance.start) / totalSeconds) * width);
        ctx.beginPath();
        ctx.roundRect(x, y, w, laneHeight, 2);
        ctx.fill();
      }
    });
  }
}
