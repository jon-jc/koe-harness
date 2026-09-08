/**
 * User preferences.
 *
 * Everything here is a *per-device* choice — which microphone, how sensitive
 * the endpointer is, whether the UI is dark. None of it belongs on the server:
 * the same account on a laptop and in a meeting room wants different answers,
 * and a preference that follows you onto the wrong hardware is worse than one
 * that does not follow you at all. API keys are the deliberate exception and
 * live server-side, because the browser is the wrong place to hold a secret.
 *
 * Loading is defensive in a way that looks paranoid and is not. This blob
 * survives across versions of the app, so it will eventually be read by code
 * that did not write it: a field may be missing, renamed, or hold whatever a
 * future bug put there. Every value is therefore range-checked on the way in,
 * and a bad one degrades to the default rather than reaching an AudioContext
 * or a slider as `NaN`.
 */

export type AudioSource = "microphone" | "system" | "both";

export interface Prefs {
  /** Where audio comes from. */
  readonly source: AudioSource;
  /** `deviceId` from enumerateDevices, or "" for the system default. */
  readonly inputDeviceId: string;
  /** Software gain applied after capture, 0.25–4. */
  readonly gain: number;
  /** Browser-side audio processing. Off is right for music or a room mic. */
  readonly echoCancellation: boolean;
  readonly noiseSuppression: boolean;
  readonly autoGainControl: boolean;
  /** How often interim recognition runs, ms. Lower is snappier and dearer. */
  readonly partialIntervalMs: number;
  /** Silence after speech before an utterance is closed, ms. */
  readonly silenceToEndMs: number;
  /** dB above the noise floor for a frame to count as speech. */
  readonly speechThresholdDb: number;
  /** Empty means "use the language default", which differs JA vs EN. */
  readonly useEndpointDefaults: boolean;
}

export const DEFAULTS: Prefs = {
  source: "microphone",
  inputDeviceId: "",
  gain: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  partialIntervalMs: 400,
  silenceToEndMs: 900,
  speechThresholdDb: 9,
  useEndpointDefaults: true,
};

/** Bounds shared by the sliders and by the server's own clamping. */
export const LIMITS = {
  gain: { min: 0.25, max: 4, step: 0.05 },
  partialIntervalMs: { min: 150, max: 2000, step: 50 },
  silenceToEndMs: { min: 200, max: 3000, step: 50 },
  speechThresholdDb: { min: 3, max: 24, step: 1 },
} as const;

const KEY = "koe.prefs";

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function bool(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

export function load(): Prefs {
  let raw: Record<string, unknown> = {};
  try {
    raw = JSON.parse(localStorage.getItem(KEY) ?? "{}") as Record<string, unknown>;
  } catch {
    // Private browsing, blocked site data, or a corrupt blob. Defaults are a
    // working configuration; refusing to start over a preferences file is not.
  }
  if (typeof raw !== "object" || raw === null) raw = {};

  const source = raw.source;
  return {
    source:
      source === "system" || source === "both" || source === "microphone"
        ? source
        : DEFAULTS.source,
    inputDeviceId: typeof raw.inputDeviceId === "string" ? raw.inputDeviceId : "",
    gain: clamp(raw.gain, LIMITS.gain.min, LIMITS.gain.max, DEFAULTS.gain),
    echoCancellation: bool(raw.echoCancellation, DEFAULTS.echoCancellation),
    noiseSuppression: bool(raw.noiseSuppression, DEFAULTS.noiseSuppression),
    autoGainControl: bool(raw.autoGainControl, DEFAULTS.autoGainControl),
    partialIntervalMs: clamp(
      raw.partialIntervalMs,
      LIMITS.partialIntervalMs.min,
      LIMITS.partialIntervalMs.max,
      DEFAULTS.partialIntervalMs,
    ),
    silenceToEndMs: clamp(
      raw.silenceToEndMs,
      LIMITS.silenceToEndMs.min,
      LIMITS.silenceToEndMs.max,
      DEFAULTS.silenceToEndMs,
    ),
    speechThresholdDb: clamp(
      raw.speechThresholdDb,
      LIMITS.speechThresholdDb.min,
      LIMITS.speechThresholdDb.max,
      DEFAULTS.speechThresholdDb,
    ),
    useEndpointDefaults: bool(raw.useEndpointDefaults, DEFAULTS.useEndpointDefaults),
  };
}

export function save(prefs: Prefs): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    // A preference that cannot be remembered still applies to this session.
  }
}

export function reset(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* see save */
  }
}
