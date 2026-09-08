/**
 * The koe web client.
 *
 * The interface is built around one idea: **make the pipeline's reasoning
 * visible**. A transcript that simply appears tells you nothing about whether
 * to trust it. So:
 *
 *   - Settled text and the tail the model is still deciding on are styled
 *     apart, because a caption that rewrites itself is hard to read and a
 *     reader deserves to know which half is safe.
 *   - The level strip colours by whether the endpointer heard speech, which
 *     turns a config value into something you can watch working.
 *   - **Clicking a claim in the 議事録 highlights the transcript span it came
 *     from.** That is the groundedness mechanism made tangible: a citation is
 *     not an assertion about the transcript, it points at a piece of it.
 *   - The routing panel shows which backend was chosen and why, including the
 *     ones that were rejected and for what reason — and which model is
 *     actually answering right now, with the way to change it one click away.
 *
 * Rendering is hand-written DOM rather than a framework, and the hot paths
 * (level meter, canvases) bypass application state entirely — see `state.ts`
 * for why that is a deliberate constraint on an audio page.
 */

import "./styles.css";

import { MicrophoneCapture } from "./audio";
import { copyText, el, h } from "./dom";
import { strings, type Strings, type UILang } from "./i18n";
import { SettingsDialog, ShortcutsDialog } from "./settings";
import {
  INITIAL,
  Store,
  speakerColorIndex,
  type ASRLang,
  type Claim,
  type MinutesView,
  type ProviderRow,
  type Utterance,
} from "./state";
import { StreamClient, type MinutesPayload, type ServerMessage } from "./stream-client";
import { Notifications } from "./toast";
import { LevelStrip, SpeakerTimeline } from "./viz";

/* ------------------------------------------------------------------ utils */

/**
 * Loose normalization for citation matching.
 *
 * A deliberately simplified mirror of the server's scoring normalizer: fold
 * width and case, drop punctuation and whitespace. It only has to be good
 * enough to locate a highlight — the authoritative verification already
 * happened server-side, and a missed highlight degrades to highlighting the
 * whole utterance rather than to a wrong answer.
 */
function loose(text: string): string {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\s　]/g, "")
    .replace(/[、。「」『』（）()！？!?,.:;・~〜ー\-—"'']/g, "");
}

interface Citation {
  index: number;
  raw: string | null;
}

/** Locate the utterance a quote came from. */
function findCitation(utterances: readonly Utterance[], quote: string): Citation | null {
  const needle = loose(quote);
  if (!needle) return null;
  for (let i = 0; i < utterances.length; i++) {
    if (loose(utterances[i].text).includes(needle)) {
      // Try an exact raw match too, so the highlight can be a span rather than
      // the whole line. Falling back to the whole line is fine.
      const raw = utterances[i].text.includes(quote) ? quote : null;
      return { index: i, raw };
    }
  }
  return null;
}

/**
 * Search folding: width and case, nothing else.
 *
 * Deliberately weaker than `loose` — a searcher who types a punctuation mark
 * means it, whereas a citation is machine-generated and only has to be
 * *found*. Width folding stays because ＡＢＣ and ABC are the same query to
 * everyone except a computer, and Japanese input methods produce both.
 */
function fold(text: string): string {
  return text.normalize("NFKC").toLowerCase();
}

function matchesQuery(utterance: Utterance, folded: string): boolean {
  return (
    fold(utterance.text).includes(folded) || fold(utterance.speaker).includes(folded)
  );
}

/**
 * Append `text` to `host`, wrapping occurrences of `needle` in `<mark>`.
 *
 * Positions are taken from a plain lowercase fold rather than NFKC, because
 * NFKC can change string *length* and the offsets would then point at the
 * wrong characters. When lowering does change the length — a handful of
 * locale-specific characters do — the highlight is skipped rather than
 * misplaced. The line is still shown; it is only less decorated.
 */
function appendHighlighted(host: HTMLElement, text: string, needle: string): void {
  const hay = text.toLowerCase();
  const pin = needle.toLowerCase();
  if (!pin || hay.length !== text.length || !hay.includes(pin)) {
    host.append(text);
    return;
  }
  let at = 0;
  for (;;) {
    const found = hay.indexOf(pin, at);
    if (found === -1) break;
    if (found > at) host.append(text.slice(at, found));
    host.append(h("mark", { class: "find", text: text.slice(found, found + pin.length) }));
    at = found + pin.length;
  }
  if (at < text.length) host.append(text.slice(at));
}

function fmtDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** True when a keystroke belongs to whatever the user is typing into. */
function isTyping(): boolean {
  const active = document.activeElement;
  return (
    active instanceof HTMLInputElement ||
    active instanceof HTMLTextAreaElement ||
    active instanceof HTMLSelectElement ||
    (active instanceof HTMLElement && active.isContentEditable)
  );
}

/* ------------------------------------------------------------------ theme */

type Theme = "light" | "dark" | "system";

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("koe.theme", theme);
  } catch {
    // Private browsing, or site data blocked. A remembered preference is a
    // convenience; losing it must not break the page.
  }
}

function loadTheme(): Theme {
  try {
    const stored = localStorage.getItem("koe.theme");
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    /* see applyTheme */
  }
  return "system";
}

/* ------------------------------------------------------------------ app */

class App {
  private readonly store = new Store({ ...INITIAL, uiLang: detectUILang() });
  private client: StreamClient | null = null;
  private capture: MicrophoneCapture | null = null;
  private theme: Theme = loadTheme();
  private tickTimer = 0;
  private startedAt = 0;
  private shownError: string | null = null;
  /** Demo playback is server-driven, so there is no capture to tear down. */
  private demoMode = false;

  private readonly levelStrip = new LevelStrip(el<HTMLCanvasElement>("level-canvas"));
  private readonly timeline = new SpeakerTimeline(el<HTMLCanvasElement>("timeline-canvas"));

  private readonly refs = {
    toggle: el<HTMLButtonElement>("toggle"),
    toggleLabel: el<HTMLSpanElement>("toggle-label"),
    demo: el<HTMLButtonElement>("demo"),
    demoLabel: el<HTMLSpanElement>("demo-label"),
    statusChip: el<HTMLSpanElement>("status-chip"),
    speechChip: el<HTMLSpanElement>("speech-chip"),
    backend: el<HTMLSpanElement>("backend"),
    levelFill: el<HTMLDivElement>("level-fill"),
    transcript: el<HTMLDivElement>("transcript"),
    side: el<HTMLDivElement>("side"),
    stats: el<HTMLDivElement>("stats"),
    asrLang: el<HTMLSelectElement>("asr-lang"),
    uiLang: el<HTMLSelectElement>("ui-lang"),
    themeBtn: el<HTMLButtonElement>("theme"),
    settingsBtn: el<HTMLButtonElement>("settings"),
    helpBtn: el<HTMLButtonElement>("help"),
    tabs: el<HTMLDivElement>("tabs"),
    live: el<HTMLDivElement>("live-region"),
    search: el<HTMLDivElement>("search"),
    searchInput: el<HTMLInputElement>("search-input"),
    searchCount: el<HTMLSpanElement>("search-count"),
    searchClose: el<HTMLButtonElement>("search-close"),
    searchToggle: el<HTMLButtonElement>("search-toggle"),
    copyTranscript: el<HTMLButtonElement>("copy-transcript"),
  };

  private readonly notify = new Notifications(undefined, this.refs.live);
  private readonly settings = new SettingsDialog(this.notify, strings(detectUILang()), () =>
    void this.loadActiveModel(),
  );
  private readonly shortcuts = new ShortcutsDialog(strings(detectUILang()));

  constructor() {
    applyTheme(this.theme);
    this.bind();
    this.store.subscribe(() => this.render());
    this.render();
    this.levelStrip.render();
    this.timeline.render([], 0);
    void this.loadProviders();
    void this.loadActiveModel();

    const redraw = () => {
      this.levelStrip.render();
      const state = this.store.get();
      this.timeline.render(state.utterances, Math.max(state.durationS, 1));
    };
    window.addEventListener("resize", redraw);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  }

  private get s(): Strings {
    return strings(this.store.get().uiLang);
  }

  /* ---------------------------------------------------------------- wiring */

  private bind(): void {
    this.refs.toggle.addEventListener("click", () => void this.toggle());
    this.refs.demo.addEventListener("click", () => void this.startDemo());

    this.refs.asrLang.addEventListener("change", () => {
      this.store.set({ asrLang: this.refs.asrLang.value as ASRLang });
    });

    this.refs.uiLang.addEventListener("change", () => {
      const uiLang = this.refs.uiLang.value as UILang;
      document.documentElement.lang = uiLang;
      this.settings.setStrings(strings(uiLang));
      this.shortcuts.setStrings(strings(uiLang));
      this.store.set({ uiLang });
    });

    this.refs.themeBtn.addEventListener("click", () => {
      const order: Theme[] = ["system", "light", "dark"];
      this.theme = order[(order.indexOf(this.theme) + 1) % order.length];
      applyTheme(this.theme);
      this.refs.themeBtn.textContent = this.theme === "dark" ? "◐" : this.theme === "light" ? "○" : "◑";
      this.levelStrip.render();
      const state = this.store.get();
      this.timeline.render(state.utterances, Math.max(state.durationS, 1));
    });

    this.refs.settingsBtn.addEventListener("click", () => void this.settings.open());
    this.refs.helpBtn.addEventListener("click", () => this.shortcuts.open());

    this.refs.tabs.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-tab]");
      if (!target) return;
      this.store.set({ tab: target.dataset.tab as "minutes" | "routing" | "metrics" });
    });

    // Arrow keys move between tabs, which is what the tablist role promises
    // and what a keyboard user will try. Without it the role is a lie.
    this.refs.tabs.addEventListener("keydown", (event) => {
      const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (step === 0) return;
      event.preventDefault();
      const buttons = [...this.refs.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]")];
      const at = buttons.findIndex((b) => b.getAttribute("aria-selected") === "true");
      const next = buttons[(at + step + buttons.length) % buttons.length];
      this.store.set({ tab: next.dataset.tab as "minutes" | "routing" | "metrics" });
      next.focus();
    });

    /* -- search ---------------------------------------------------------- */

    this.refs.searchToggle.addEventListener("click", () => this.openSearch());
    this.refs.searchClose.addEventListener("click", () => this.closeSearch());
    this.refs.searchInput.addEventListener("input", () => {
      this.store.set({ query: this.refs.searchInput.value });
    });
    this.refs.searchInput.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        this.closeSearch();
      }
    });

    this.refs.copyTranscript.addEventListener("click", () => void this.copyTranscript());

    /* -- shortcuts ------------------------------------------------------- */

    document.addEventListener("keydown", (event) => {
      // A modal owns the keyboard while it is open; the browser already gives
      // it Escape and a focus trap, and a global handler firing behind it
      // would act on a surface the user cannot see.
      if (document.querySelector("dialog[open]")) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const typing = isTyping();
      const inControl =
        typing ||
        document.activeElement instanceof HTMLButtonElement;

      // Space toggles recording, unless the user is in a control — where space
      // means "activate this button" and stealing it would be hostile.
      if (event.code === "Space" && !inControl) {
        event.preventDefault();
        void this.toggle();
        return;
      }

      if (event.key === "Escape") {
        if (this.store.get().query) this.closeSearch();
        else this.store.set({ selectedQuote: null });
        return;
      }

      if (typing) return;

      if (event.key === "/") {
        event.preventDefault();
        this.openSearch();
      } else if (event.key === ",") {
        event.preventDefault();
        void this.settings.open();
      } else if (event.key === "?") {
        event.preventDefault();
        this.shortcuts.open();
      }
    });
  }

  private openSearch(): void {
    this.refs.search.hidden = false;
    this.refs.searchToggle.setAttribute("aria-expanded", "true");
    this.refs.searchInput.focus();
    this.refs.searchInput.select();
  }

  private closeSearch(): void {
    this.refs.search.hidden = true;
    this.refs.searchToggle.setAttribute("aria-expanded", "false");
    this.refs.searchInput.value = "";
    this.store.set({ query: "" });
  }

  /* ---------------------------------------------------------------- session */

  private async toggle(): Promise<void> {
    const status = this.store.get().status;
    if (status === "live" || status === "connecting") await this.stop();
    else await this.start();
  }

  /**
   * Play a scripted meeting without touching the microphone.
   *
   * The server synthesizes audio and pushes it through the real session, so
   * this exercises endpointing, stabilization and the ASR call rather than
   * replaying recorded events. It exists because most people who open this
   * will not grant a microphone prompt, and the realtime path is the part
   * worth seeing.
   */
  private async startDemo(): Promise<void> {
    if (this.store.get().status === "live") return;
    this.demoMode = true;
    this.store.set({
      status: "connecting",
      utterances: [],
      committed: "",
      pending: "",
      minutes: null,
      selectedQuote: null,
      error: null,
      durationS: 0,
      costUsd: 0,
    });
    this.levelStrip.clear();

    const client = this.connect();
    try {
      await client.connect();
    } catch {
      this.demoMode = false;
      this.fail(this.s.cannotConnect);
      return;
    }

    const meeting = this.store.get().asrLang === "en" ? "quarterly-en" : "quarterly-ja";
    client.startDemo(meeting, this.store.get().asrLang);
    this.startedAt = performance.now();
    this.levelStrip.start();
    this.store.set({ status: "live" });
    this.tickTimer = window.setInterval(() => {
      this.store.set({ durationS: (performance.now() - this.startedAt) / 1000 });
    }, 250);
  }

  /** Build a client bound to this app's handlers. */
  private connect(): StreamClient {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const client = new StreamClient(`${scheme}://${location.host}/v1/stream`, {
      onMessage: (message) => this.onMessage(message),
      onError: () => this.fail(this.s.cannotConnect),
      onClose: () => {
        if (this.store.get().status === "live") this.store.set({ status: "stopped" });
      },
    });
    this.client = client;
    return client;
  }

  private async start(): Promise<void> {
    this.store.set({
      status: "connecting",
      utterances: [],
      committed: "",
      pending: "",
      minutes: null,
      selectedQuote: null,
      error: null,
      durationS: 0,
      costUsd: 0,
      framesSent: 0,
      framesDropped: 0,
    });
    this.levelStrip.clear();

    this.demoMode = false;
    const client = this.connect();

    try {
      await client.connect();
    } catch {
      this.fail(this.s.cannotConnect);
      return;
    }

    client.start({ language: this.store.get().asrLang, partialIntervalMs: 400 });

    this.capture = new MicrophoneCapture({
      onFrame: (frame) => client.sendAudio(frame),
      onLevel: (level) => {
        this.refs.levelFill.style.width = `${Math.min(100, level * 140)}%`;
        this.levelStrip.push(level);
      },
    });

    try {
      await this.capture.start();
    } catch {
      // Overwhelmingly this is a denied permission prompt; the browser gives
      // no way to distinguish that from a missing device.
      this.fail(this.s.micDenied);
      client.close();
      this.capture = null;
      return;
    }

    this.startedAt = performance.now();
    this.levelStrip.start();
    this.store.set({ status: "live" });
    this.tickTimer = window.setInterval(() => {
      this.store.set({ durationS: (performance.now() - this.startedAt) / 1000 });
    }, 250);
  }

  private async stop(): Promise<void> {
    window.clearInterval(this.tickTimer);
    if (this.demoMode) {
      // The server owns demo playback; closing the socket cancels it.
      this.demoMode = false;
      this.levelStrip.stop();
      this.client?.close();
      this.store.set({ status: "stopped", speaking: false, pending: "" });
      return;
    }
    this.levelStrip.stop();
    this.refs.levelFill.style.width = "0%";
    this.store.set({ status: "stopping", speaking: false, pending: "" });

    await this.capture?.stop();
    this.capture = null;
    this.client?.stop();

    this.store.set({ status: "stopped" });
  }

  private requestMinutes(): void {
    if (!this.client?.connected) return;
    this.store.set({ minutesPending: true, tab: "minutes" });
    this.client.requestMinutes();
  }

  private fail(message: string): void {
    this.store.set({ status: "error", error: message });
  }

  /* ---------------------------------------------------------------- events */

  private onMessage(message: ServerMessage): void {
    switch (message.type) {
      case "started":
        this.store.set({ provider: message.provider });
        break;

      case "speech":
        this.levelStrip.setSpeaking(message.state === "start");
        this.store.set({ speaking: message.state === "start" });
        break;

      case "partial":
        this.store.set({ committed: message.committed, pending: message.pending });
        break;

      case "final": {
        const utterance: Utterance = {
          text: message.text,
          start: message.start,
          end: message.end,
          speaker: message.speaker || "—",
        };
        this.store.set({
          utterances: [...this.store.get().utterances, utterance],
          committed: "",
          pending: "",
        });
        // Announce finals to assistive technology; interim hypotheses would
        // make a screen reader unusable by re-reading a churning line.
        this.refs.live.setAttribute("aria-live", "polite");
        this.refs.live.textContent = `${utterance.speaker}: ${utterance.text}`;
        break;
      }

      case "transcript":
        this.store.set({
          durationS: message.duration,
          costUsd: message.cost_usd,
          framesSent: this.client?.sentFrames ?? 0,
          framesDropped: this.client?.droppedFrames ?? 0,
        });
        break;

      case "minutes":
        this.store.set({
          minutesPending: false,
          minutes: toMinutesView(message.minutes, {
            grounded: message.grounded,
            totalClaims: message.total_claims,
            dropped: message.dropped,
            repairs: message.repairs,
            costUsd: message.cost_usd,
          }),
        });
        break;

      case "level":
        // Demo playback has no local microphone, so the level comes from the
        // server reporting the amplitude it actually generated.
        this.refs.levelFill.style.width = `${Math.min(100, message.value * 140)}%`;
        this.levelStrip.push(message.value);
        break;

      case "demo_finished":
        window.clearInterval(this.tickTimer);
        this.levelStrip.stop();
        this.demoMode = false;
        this.store.set({ status: "stopped", speaking: false, pending: "" });
        break;

      case "error":
        this.store.set({ minutesPending: false });
        this.fail(
          message.message.includes("capacity") ? this.s.atCapacity : message.message,
        );
        break;
    }
  }

  private async loadProviders(): Promise<void> {
    try {
      const response = await fetch("/v1/providers");
      if (!response.ok) return;
      const body = (await response.json()) as {
        ranked: Array<{
          name: string;
          score: number;
          cost_per_audio_minute_usd: number;
          expected_error_rate: number;
          measured: boolean;
        }>;
        rejected: Record<string, string>;
      };
      this.store.set({
        providers: body.ranked.map(
          (row): ProviderRow => ({
            name: row.name,
            score: row.score,
            costPerAudioMinuteUsd: row.cost_per_audio_minute_usd,
            expectedErrorRate: row.expected_error_rate,
            measured: row.measured,
          }),
        ),
        rejected: body.rejected ?? {},
      });
    } catch {
      // The panel is informational; a failed fetch leaves it empty rather than
      // taking the page down.
    }
  }

  /** Which LLM is answering right now — refreshed whenever a key changes. */
  private async loadActiveModel(): Promise<void> {
    try {
      const response = await fetch("/v1/credentials");
      if (!response.ok) return;
      const body = (await response.json()) as { active_llm: string };
      this.store.set({ activeLlm: body.active_llm });
    } catch {
      /* informational only — see loadProviders */
    }
  }

  /* ---------------------------------------------------------------- copy */

  private async copyTranscript(): Promise<void> {
    const state = this.store.get();
    if (state.utterances.length === 0) return;
    const text = state.utterances
      .map((u) => `[${fmtDuration(u.start)}] ${u.speaker}: ${u.text}`)
      .join("\n");
    this.report(await copyText(text));
  }

  private async copyMinutes(): Promise<void> {
    const minutes = this.store.get().minutes;
    if (!minutes) return;
    this.report(await copyText(minutesToText(minutes, this.s)));
  }

  private report(ok: boolean): void {
    if (ok) this.notify.ok(this.s.copied);
    else this.notify.error(this.s.copyFailed);
  }

  /* ---------------------------------------------------------------- render */

  private render(): void {
    const state = this.store.get();
    const s = this.s;
    const live = state.status === "live";

    this.refs.toggleLabel.textContent = live ? s.stop : s.record;
    this.refs.demoLabel.textContent = s.demo;
    this.refs.demo.title = s.demoHint;
    this.refs.demo.disabled = live || state.status === "connecting";
    this.refs.toggle.classList.toggle("recording", live);
    this.refs.toggle.setAttribute("aria-pressed", String(live));

    const statusText =
      state.status === "live"
        ? s.listening
        : state.status === "connecting"
          ? s.connecting
          : state.status === "stopped" || state.status === "stopping"
            ? s.stopped
            : state.status === "error"
              ? "error"
              : s.idle;
    this.refs.statusChip.textContent = statusText;
    this.refs.statusChip.className = `chip${live ? " live" : state.status === "error" ? " bad" : ""}`;

    this.refs.speechChip.textContent = state.speaking ? s.speaking : s.silence;
    this.refs.speechChip.className = `chip${state.speaking ? " live" : ""}`;
    this.refs.backend.textContent = state.provider || "—";

    this.refs.uiLang.value = state.uiLang;
    this.refs.asrLang.value = state.asrLang;
    this.refs.themeBtn.title = s.theme;
    this.refs.themeBtn.setAttribute("aria-label", s.theme);

    this.renderLabels(s);
    this.renderTranscript(s);
    this.renderSide(s);
    this.renderStats(s);
    this.timeline.render(state.utterances, Math.max(state.durationS, 1));

    // One notification per distinct error, not one per render — the store
    // notifies on every frame that touches it.
    if (state.error && state.error !== this.shownError) {
      this.shownError = state.error;
      this.notify.error(state.error);
    } else if (!state.error) {
      this.shownError = null;
    }
  }

  private renderLabels(s: Strings): void {
    el("t-transcript").textContent = s.transcript;
    el("t-level").textContent = s.level;
    el("t-speakers").textContent = s.speakers;
    el("tab-minutes").textContent = s.minutes;
    el("tab-routing").textContent = s.routing;
    el("tab-metrics").textContent = s.metrics;
    el("t-ui-lang").textContent = s.uiLanguage;
    el("t-asr-lang").textContent = s.asrLanguage;
    el("t-tagline").textContent = s.tagline;
    el("t-hint").replaceChildren(h("kbd", { text: "Space" }), ` ${s.toggleRecord}`);

    const auto = this.refs.asrLang.querySelector('option[value="unknown"]');
    if (auto) auto.textContent = s.auto;

    this.refs.settingsBtn.title = s.settings;
    this.refs.settingsBtn.setAttribute("aria-label", s.settings);
    this.refs.helpBtn.title = s.shortcuts;
    this.refs.helpBtn.setAttribute("aria-label", s.shortcuts);
    this.refs.searchToggle.title = s.search;
    this.refs.searchToggle.setAttribute("aria-label", s.search);
    this.refs.searchClose.setAttribute("aria-label", s.close);
    this.refs.copyTranscript.title = s.copyTranscript;
    this.refs.copyTranscript.setAttribute("aria-label", s.copyTranscript);
    this.refs.searchInput.placeholder = s.searchPlaceholder;
    this.refs.searchInput.setAttribute("aria-label", s.searchPlaceholder);
  }

  private renderTranscript(s: Strings): void {
    const state = this.store.get();
    const { utterances, committed, pending, selectedQuote, query } = state;
    const host = this.refs.transcript;
    host.replaceChildren();

    const needle = query.trim();
    const folded = fold(needle);
    const searching = needle.length > 0;

    if (utterances.length === 0 && !committed && !pending) {
      host.append(h("div", { class: "empty" }, h("span", { class: "k", text: "声" }), s.emptyTranscript));
      this.refs.searchCount.textContent = "";
      this.refs.copyTranscript.disabled = true;
      return;
    }
    this.refs.copyTranscript.disabled = false;

    const citation = selectedQuote ? findCitation(utterances, selectedQuote) : null;
    let hits = 0;

    utterances.forEach((utterance, index) => {
      const hit = !searching || matchesQuery(utterance, folded);
      if (hit && searching) hits += 1;

      const speaker = utterance.speaker || "—";
      const row = h("div", { class: "utt" });
      row.style.setProperty("--spk", `var(--spk-${speakerColorIndex(speaker)})`);
      // Dimmed rather than removed: a transcript is a sequence, and dropping
      // the misses would hide that a match sits between two other exchanges.
      if (searching && !hit) row.classList.add("dimmed");
      if (citation?.index === index) {
        row.classList.add("cited");
        row.id = "cited-utterance";
      }

      row.append(h("div", { class: "who", text: speaker }));

      const said = h("div", { class: "said ja" });
      if (citation?.index === index && citation.raw) {
        // Highlight the exact cited span when it survives verbatim.
        const at = utterance.text.indexOf(citation.raw);
        said.append(
          utterance.text.slice(0, at),
          h("mark", { class: "hit", text: citation.raw }),
          utterance.text.slice(at + citation.raw.length),
        );
      } else if (searching && hit) {
        appendHighlighted(said, utterance.text, needle);
      } else {
        said.textContent = utterance.text;
      }
      said.append(h("span", { class: "at", text: fmtDuration(utterance.start) }));
      row.append(said);
      host.append(row);
    });

    if (committed || pending) {
      const row = h("div", { class: "interim" });
      row.append(h("div", { class: "who" }));
      const said = h("div", { class: "said ja" });
      said.append(committed);
      if (pending) said.append(h("span", { class: "pending", text: pending }));
      row.append(said);
      host.append(row);
    }

    this.refs.searchCount.textContent = searching
      ? hits > 0
        ? `${hits} ${hits === 1 ? s.match : s.matches}`
        : s.noMatches
      : "";

    // Do not yank the view while someone is reading search results.
    if (!searching) {
      const target = host.querySelector("#cited-utterance") ?? host.lastElementChild;
      target?.scrollIntoView({ block: "nearest" });
    } else {
      host.querySelector(".utt:not(.dimmed)")?.scrollIntoView({ block: "nearest" });
    }
  }

  private renderSide(s: Strings): void {
    const state = this.store.get();
    const host = this.refs.side;
    host.replaceChildren();

    for (const button of this.refs.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
      const selected = button.dataset.tab === state.tab;
      button.setAttribute("aria-selected", String(selected));
      // Roving tabindex: one stop for the whole tablist, arrows move within.
      button.tabIndex = selected ? 0 : -1;
    }

    if (state.tab === "routing") return this.renderRouting(host, s);
    if (state.tab === "metrics") return this.renderMetrics(host, s);
    return this.renderMinutes(host, s);
  }

  private renderMinutes(host: HTMLElement, s: Strings): void {
    const state = this.store.get();

    if (!state.minutes) {
      const canGenerate = state.utterances.length > 0 && state.status !== "live";
      const empty = h("div", { class: "empty" }, h("span", { class: "k", text: "議" }), s.emptyMinutes);
      host.append(empty);
      const action = h("div", { style: "text-align:center;margin-top:12px" });
      const button = h("button", { class: "btn" });
      button.textContent = state.minutesPending ? s.generating : s.generate;
      button.disabled = !canGenerate || state.minutesPending;
      button.addEventListener("click", () => this.requestMinutes());
      action.append(button);
      if (!canGenerate) {
        action.append(h("p", { class: "note", text: s.emptyMinutesHint, style: "margin-top:10px" }));
      }
      host.append(action);
      return;
    }

    const m = state.minutes;
    const wrap = h("div", { class: "minutes" });

    const heading = h("div", { class: "row", style: "margin-bottom:10px" });
    if (m.title) heading.append(h("h1", { class: "ja", text: m.title, style: "margin:0" }));
    heading.append(h("div", { style: "flex:1" }));
    const copy = h("button", {
      class: "icon-btn",
      type: "button",
      title: s.copyMinutes,
      "aria-label": s.copyMinutes,
      text: "⧉",
    });
    copy.addEventListener("click", () => void this.copyMinutes());
    heading.append(copy);
    wrap.append(heading);

    const badges = h("div", { style: "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px" });
    badges.append(
      h("span", {
        class: m.grounded === m.totalClaims ? "chip live" : "chip warn",
        text: `${s.verified} ${m.grounded}/${m.totalClaims} ${s.claims}`,
      }),
    );
    if (m.repairs > 0) badges.append(h("span", { class: "chip", text: `${s.repairs} ${m.repairs}` }));
    if (m.dropped.length > 0) {
      badges.append(h("span", { class: "chip bad", text: `${s.dropped} ${m.dropped.length}` }));
    }
    wrap.append(badges);

    if (m.participants.length > 0) {
      const row = h("div", { style: "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px" });
      for (const person of m.participants) {
        const chip = h("span", { class: "chip", text: person });
        chip.style.color = `var(--spk-${speakerColorIndex(person)})`;
        row.append(chip);
      }
      wrap.append(row);
    }

    if (m.summary) {
      wrap.append(h("h2", { text: s.summary }), h("p", { class: "ja", text: m.summary }));
    }

    if (m.topics.length > 0) {
      wrap.append(h("h2", { text: s.topics }));
      for (const topic of m.topics) {
        wrap.append(
          h("p", { class: "ja" }, h("b", { text: topic.title }), " — ", topic.summary),
        );
      }
    }

    const decisions = m.claims.filter((c) => c.kind === "decision");
    const actions = m.claims.filter((c) => c.kind === "action");

    if (decisions.length > 0) {
      wrap.append(h("h2", { text: s.decisions }));
      for (const claim of decisions) wrap.append(this.claimNode(claim, s));
    }

    if (actions.length > 0) {
      wrap.append(h("h2", { text: s.actions }));
      for (const claim of actions) wrap.append(this.claimNode(claim, s));
    }

    if (m.dropped.length > 0) {
      wrap.append(h("h2", { text: s.dropped }));
      for (const text of m.dropped) {
        const node = h("div", { class: "claim", style: "cursor:default;opacity:0.72" });
        node.append(
          h("s", { class: "ja", text }),
          h("div", { class: "row" }, h("span", { text: s.droppedHint })),
        );
        wrap.append(node);
      }
    }

    host.append(wrap);
  }

  /** One claim, clickable to reveal its source in the transcript. */
  private claimNode(claim: Claim, s: Strings): HTMLElement {
    const state = this.store.get();
    const selected = state.selectedQuote === claim.quote;

    const node = h("button", { class: "claim", type: "button" });
    node.setAttribute("aria-pressed", String(selected));

    if (claim.kind === "action") {
      node.append(
        h("div", { class: "task" }, h("span", { class: "box" }), h("span", { text: claim.text })),
      );
    } else {
      node.append(h("span", { text: claim.text }));
    }

    const row = h("div", { class: "row" });
    if (claim.kind === "action") {
      const owner = h("span", { class: "owner", text: claim.owner || s.unassigned });
      if (claim.owner) owner.style.setProperty("--spk", `var(--spk-${speakerColorIndex(claim.owner)})`);
      row.append(h("span", { text: `${s.owner}:` }), owner);
      if (claim.due) row.append(h("span", { text: `· ${s.due}: ${claim.due}` }));
    } else if (claim.speaker) {
      row.append(h("span", { text: claim.speaker }));
    }
    if (claim.quote) row.append(h("span", { class: "cite", text: s.citation }));
    node.append(row);

    node.addEventListener("click", () => {
      // Selecting a citation while filtering would highlight a line the filter
      // is hiding, so clear the filter and show the source.
      this.store.set({ selectedQuote: selected ? null : claim.quote, query: "" });
      if (!selected) this.closeSearch();
    });
    return node;
  }

  private renderRouting(host: HTMLElement, s: Strings): void {
    const state = this.store.get();

    /*
     * The routing table explains ASR backend selection; the LLM behind the
     * 議事録 is a separate decision, and until it was surfaced here people
     * assumed the mock output was the model's. Naming it — and putting the
     * way to change it one click away — is the difference between a demo and
     * something someone can actually point at their own account.
     */
    const usingMock = !state.activeLlm || state.activeLlm.startsWith("mock");
    const llm = h("div", { class: "provider" });
    const llmHead = h("div", { class: "provider-head" });
    llmHead.append(
      h("span", { class: "provider-name", text: s.activeModel }),
      h("span", {
        class: usingMock ? "chip warn" : "chip live",
        text: usingMock ? s.usingMock : state.activeLlm,
      }),
      h("div", { style: "flex:1" }),
    );
    const open = h("button", { class: "btn ghost", type: "button", text: s.settings });
    open.addEventListener("click", () => void this.settings.open());
    llmHead.append(open);
    llm.append(llmHead);
    llm.append(
      h("p", {
        class: "note",
        style: "margin:0",
        text: usingMock ? s.usingMockHint : s.apiKeysHint,
      }),
    );
    host.append(llm);

    host.append(h("p", { class: "note", text: s.routingHint, style: "margin:14px 0" }));

    if (state.providers.length === 0) {
      host.append(h("div", { class: "empty", text: s.noSession }));
      return;
    }

    const table = h("table", { class: "tbl" });
    table.append(
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", { text: s.provider }),
          h("th", { class: "num", text: s.score }),
          h("th", { class: "num", text: s.errorRate }),
          h("th", { class: "num", text: s.costPerMin }),
          h("th", { text: s.state }),
        ),
      ),
    );

    const body = h("tbody");
    state.providers.forEach((row, index) => {
      const tr = h("tr");
      const name = h("td", {}, h("b", { text: row.name }));
      if (index === 0) {
        name.append(" ", h("span", { class: "chip live", text: s.chosen }));
      }
      tr.append(
        name,
        h("td", { class: `num${index === 0 ? " win" : ""}`, text: row.score.toFixed(3) }),
        h("td", { class: "num", text: `${(row.expectedErrorRate * 100).toFixed(1)}%` }),
        h("td", { class: "num", text: `$${row.costPerAudioMinuteUsd.toFixed(4)}` }),
        h("td", {}, h("span", { class: "chip", text: row.measured ? s.measured : s.prior })),
      );
      body.append(tr);
    });
    table.append(body);
    host.append(table);

    const rejected = Object.entries(state.rejected);
    if (rejected.length > 0) {
      host.append(h("h2", { class: "pane-title", style: "margin-top:20px", text: "rejected" }));
      for (const [name, reason] of rejected) {
        host.append(h("p", { class: "note" }, h("b", { text: name }), ` — ${reason}`));
      }
    }
  }

  private renderMetrics(host: HTMLElement, s: Strings): void {
    const state = this.store.get();
    const rows: Array<[string, string]> = [
      [s.audio, fmtDuration(state.durationS)],
      [s.segments, String(state.utterances.length)],
      [s.speakers, String(new Set(state.utterances.map((u) => u.speaker)).size)],
      [s.cost, `$${state.costUsd.toFixed(5)}`],
      [s.frames, String(state.framesSent)],
      [s.framesDropped, String(state.framesDropped)],
    ];

    const table = h("table", { class: "tbl" });
    const body = h("tbody");
    for (const [label, value] of rows) {
      body.append(h("tr", {}, h("td", { text: label }), h("td", { class: "num", text: value })));
    }
    table.append(body);
    host.append(table);

    if (state.minutes) {
      host.append(h("h2", { class: "pane-title", style: "margin-top:20px", text: s.minutes }));
      const t2 = h("table", { class: "tbl" });
      const b2 = h("tbody");
      b2.append(
        h(
          "tr",
          {},
          h("td", { text: s.verified }),
          h("td", {
            class: "num",
            text: `${state.minutes.grounded}/${state.minutes.totalClaims}`,
          }),
        ),
        h(
          "tr",
          {},
          h("td", { text: s.dropped }),
          h("td", { class: "num", text: String(state.minutes.dropped.length) }),
        ),
      );
      t2.append(b2);
      host.append(t2);
    }
  }

  private renderStats(s: Strings): void {
    const state = this.store.get();
    const parts = [
      `${fmtDuration(state.durationS)} ${s.audio}`,
      `${state.utterances.length} ${s.segments}`,
    ];
    if (state.costUsd > 0) parts.push(`$${state.costUsd.toFixed(5)}`);
    if (state.framesDropped > 0) parts.push(`${state.framesDropped} ${s.framesDropped}`);
    this.refs.stats.textContent = parts.join("  ·  ");
  }
}

/* ------------------------------------------------------------------ adapt */

/**
 * Minutes as plain text, for pasting into whatever the team actually uses.
 *
 * Markdown-shaped because Slack, Notion, Confluence and a plain mail body all
 * degrade gracefully from it, and because action items survive as checkboxes.
 */
function minutesToText(m: MinutesView, s: Strings): string {
  const lines: string[] = [];
  if (m.title) lines.push(`# ${m.title}`, "");
  if (m.participants.length > 0) lines.push(`${s.participants}: ${m.participants.join(", ")}`, "");
  if (m.summary) lines.push(`## ${s.summary}`, m.summary, "");

  if (m.topics.length > 0) {
    lines.push(`## ${s.topics}`);
    for (const topic of m.topics) lines.push(`- **${topic.title}** — ${topic.summary}`);
    lines.push("");
  }

  const decisions = m.claims.filter((c) => c.kind === "decision");
  if (decisions.length > 0) {
    lines.push(`## ${s.decisions}`);
    for (const claim of decisions) {
      lines.push(`- ${claim.text}${claim.speaker ? ` (${claim.speaker})` : ""}`);
    }
    lines.push("");
  }

  const actions = m.claims.filter((c) => c.kind === "action");
  if (actions.length > 0) {
    lines.push(`## ${s.actions}`);
    for (const claim of actions) {
      const owner = claim.owner || s.unassigned;
      const due = claim.due ? ` · ${s.due}: ${claim.due}` : "";
      lines.push(`- [ ] ${claim.text} — ${owner}${due}`);
    }
    lines.push("");
  }

  // The provenance line is the point of the whole feature: whoever receives
  // this should know how much of it was checked against the transcript.
  lines.push(`_${s.verified} ${m.grounded}/${m.totalClaims} ${s.claims}_`);
  return lines.join("\n");
}

function toMinutesView(
  payload: MinutesPayload,
  extra: {
    grounded: number;
    totalClaims: number;
    dropped: string[];
    repairs: number;
    costUsd: number;
  },
): MinutesView {
  const claims: Claim[] = [
    ...payload.decisions.map(
      (d): Claim => ({
        kind: "decision",
        text: d.statement,
        quote: d.source_quote,
        speaker: d.speaker,
        owner: "",
        due: "",
      }),
    ),
    ...payload.action_items.map(
      (a): Claim => ({
        kind: "action",
        text: a.task,
        quote: a.source_quote,
        speaker: a.speaker,
        owner: a.owner,
        due: a.due,
      }),
    ),
  ];
  return {
    title: payload.title,
    participants: payload.participants,
    summary: payload.summary,
    topics: payload.topics,
    claims,
    grounded: extra.grounded,
    totalClaims: extra.totalClaims,
    dropped: extra.dropped,
    repairs: extra.repairs,
    costUsd: extra.costUsd,
  };
}

function detectUILang(): UILang {
  return navigator.language.toLowerCase().startsWith("ja") ? "ja" : "en";
}

document.addEventListener("DOMContentLoaded", () => {
  new App();
});
