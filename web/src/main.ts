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
 *     ones that were rejected and for what reason.
 *
 * Rendering is hand-written DOM rather than a framework, and the hot paths
 * (level meter, canvases) bypass application state entirely — see `state.ts`
 * for why that is a deliberate constraint on an audio page.
 */

import "./styles.css";

import { MicrophoneCapture } from "./audio";
import { strings, type Strings, type UILang } from "./i18n";
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
import { LevelStrip, SpeakerTimeline } from "./viz";

/* ------------------------------------------------------------------ utils */

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: Array<Node | string>
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

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

function fmtDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
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
  private toastTimer = 0;
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
    tabs: el<HTMLDivElement>("tabs"),
    live: el<HTMLDivElement>("live-region"),
  };

  constructor() {
    applyTheme(this.theme);
    this.bind();
    this.store.subscribe(() => this.render());
    this.render();
    this.levelStrip.render();
    this.timeline.render([], 0);
    void this.loadProviders();

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

    this.refs.tabs.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-tab]");
      if (!target) return;
      this.store.set({ tab: target.dataset.tab as "minutes" | "routing" | "metrics" });
    });

    // Space toggles recording, unless the user is in a control — where space
    // means "activate this button" and stealing it would be hostile.
    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      const inControl =
        active instanceof HTMLButtonElement ||
        active instanceof HTMLSelectElement ||
        active instanceof HTMLInputElement;
      if (event.code === "Space" && !inControl) {
        event.preventDefault();
        void this.toggle();
      }
      if (event.key === "Escape") this.store.set({ selectedQuote: null });
    });
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
    this.renderTranscript(state.utterances, state.committed, state.pending, state.selectedQuote, s);
    this.renderSide(s);
    this.renderStats(s);
    this.timeline.render(state.utterances, Math.max(state.durationS, 1));

    if (state.error) this.toast(state.error);
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
    el("t-hint").innerHTML = `<kbd>Space</kbd> ${s.toggleRecord}`;
    const auto = this.refs.asrLang.querySelector('option[value="unknown"]');
    if (auto) auto.textContent = s.auto;
  }

  private renderTranscript(
    utterances: readonly Utterance[],
    committed: string,
    pending: string,
    selectedQuote: string | null,
    s: Strings,
  ): void {
    const host = this.refs.transcript;
    host.replaceChildren();

    if (utterances.length === 0 && !committed && !pending) {
      host.append(h("div", { class: "empty" }, h("span", { class: "k", text: "声" }), s.emptyTranscript));
      return;
    }

    const citation = selectedQuote ? findCitation(utterances, selectedQuote) : null;

    utterances.forEach((utterance, index) => {
      const speaker = utterance.speaker || "—";
      const row = h("div", { class: "utt" });
      row.style.setProperty("--spk", `var(--spk-${speakerColorIndex(speaker)})`);
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

    const target = host.querySelector("#cited-utterance") ?? host.lastElementChild;
    target?.scrollIntoView({ block: "nearest" });
  }

  private renderSide(s: Strings): void {
    const state = this.store.get();
    const host = this.refs.side;
    host.replaceChildren();

    for (const button of this.refs.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
      button.setAttribute("aria-selected", String(button.dataset.tab === state.tab));
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

    if (m.title) wrap.append(h("h1", { class: "ja", text: m.title }));

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
      this.store.set({ selectedQuote: selected ? null : claim.quote });
    });
    return node;
  }

  private renderRouting(host: HTMLElement, s: Strings): void {
    const state = this.store.get();
    host.append(h("p", { class: "note", text: s.routingHint, style: "margin-bottom:14px" }));

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

  private toast(message: string): void {
    const existing = document.querySelector(".toast");
    existing?.remove();
    const node = h("div", { class: "toast", role: "alert", text: message });
    document.body.append(node);
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => {
      node.remove();
      this.store.set({ error: null });
    }, 5000);
  }
}

/* ------------------------------------------------------------------ adapt */

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
