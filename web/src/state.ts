/**
 * Application state.
 *
 * A deliberately small observable store rather than a framework. Two reasons,
 * and the second is the real one:
 *
 *   1. Bundle size. The whole client is ~14KB gzipped; React plus a state
 *      library would be ~45KB gzipped before any application code, on a page
 *      whose job is to keep an audio pipeline fed.
 *   2. **The hot path must not go through the store at all.** The level meter
 *      updates 20 times a second and the waveform every animation frame. Those
 *      write to the DOM and the canvas directly. Routing them through a
 *      re-render — which is what a framework encourages — would spend main
 *      thread on reconciliation that competes with `AudioWorklet` message
 *      handling, and dropped input audio is unrecoverable.
 *
 * So the store holds only what changes at *human* rates: utterances, minutes,
 * connection status. Everything at frame rate bypasses it.
 */

import type { UILang } from "./i18n";

export type Status = "idle" | "connecting" | "live" | "stopping" | "stopped" | "error";
export type ASRLang = "ja" | "en" | "unknown";

export interface Utterance {
  readonly text: string;
  readonly start: number;
  readonly end: number;
  readonly speaker: string;
}

export interface Claim {
  readonly kind: "decision" | "action";
  readonly text: string;
  readonly quote: string;
  readonly speaker: string;
  readonly owner: string;
  readonly due: string;
}

export interface MinutesView {
  readonly title: string;
  readonly participants: readonly string[];
  readonly summary: string;
  readonly topics: ReadonlyArray<{ title: string; summary: string }>;
  readonly claims: readonly Claim[];
  readonly grounded: number;
  readonly totalClaims: number;
  readonly dropped: readonly string[];
  readonly repairs: number;
  readonly costUsd: number;
}

export interface ProviderRow {
  readonly name: string;
  readonly score: number;
  readonly costPerAudioMinuteUsd: number;
  readonly expectedErrorRate: number;
  readonly measured: boolean;
}

export interface AppState {
  readonly status: Status;
  readonly uiLang: UILang;
  readonly asrLang: ASRLang;
  readonly provider: string;
  readonly speaking: boolean;
  readonly utterances: readonly Utterance[];
  readonly committed: string;
  readonly pending: string;
  readonly minutes: MinutesView | null;
  readonly minutesPending: boolean;
  /** The quote of the currently selected claim; drives transcript highlighting. */
  readonly selectedQuote: string | null;
  readonly durationS: number;
  readonly costUsd: number;
  readonly framesSent: number;
  readonly framesDropped: number;
  readonly tab: "minutes" | "routing" | "metrics";
  readonly providers: readonly ProviderRow[];
  readonly rejected: Readonly<Record<string, string>>;
  /** The LLM backend actually in use, as reported by the server. */
  readonly activeLlm: string;
  /** Transcript filter. Empty means show everything. */
  readonly query: string;
  readonly error: string | null;
}

export interface AgentActivity {
  readonly busy: boolean;
  readonly step: number;
  readonly turns: number;
  readonly toolCalls: number;
  readonly compactions: number;
  readonly model: string;
}

export interface SlashCommand {
  readonly name: string;
  readonly summary: string;
  readonly usage: string;
}

export const IDLE_AGENT: AgentActivity = {
  busy: false,
  step: 0,
  turns: 0,
  toolCalls: 0,
  compactions: 0,
  model: "",
};

export const INITIAL: AppState = {
  status: "idle",
  uiLang: "ja",
  asrLang: "ja",
  provider: "",
  speaking: false,
  utterances: [],
  committed: "",
  pending: "",
  minutes: null,
  minutesPending: false,
  selectedQuote: null,
  durationS: 0,
  costUsd: 0,
  framesSent: 0,
  framesDropped: 0,
  tab: "minutes",
  providers: [],
  rejected: {},
  activeLlm: "",
  query: "",
  error: null,
};

type Listener = (state: AppState, previous: AppState) => void;

export class Store {
  private state: AppState;
  private readonly listeners = new Set<Listener>();
  private queued = false;

  constructor(initial: AppState = INITIAL) {
    this.state = initial;
  }

  get(): AppState {
    return this.state;
  }

  /**
   * Merge a patch and notify once per animation frame.
   *
   * Coalescing matters: a burst of websocket frames arriving together would
   * otherwise trigger one render each, and the renders would be identical by
   * the time the browser painted.
   */
  set(patch: Partial<AppState>): void {
    const previous = this.state;
    this.state = { ...this.state, ...patch };
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      for (const listener of this.listeners) listener(this.state, previous);
    });
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

/**
 * Assign a stable colour index to a speaker name.
 *
 * Stable across renders and across sessions for the same name, so a
 * participant does not change colour when a new speaker joins. Hashing rather
 * than first-seen ordering is what buys that.
 */
export function speakerColorIndex(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 8;
}
