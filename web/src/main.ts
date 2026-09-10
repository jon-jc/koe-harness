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
// xterm's stylesheet, imported statically even though the emulator itself is
// lazy. esbuild emits a CSS chunk for a dynamic import but nothing injects the
// <link> for it, so the terminal would open unstyled. 4 KB in the base sheet
// is the cost of that being reliable; the 280 KB of JavaScript stays lazy,
// which is the part that matters.
import "@xterm/xterm/css/xterm.css";

import { AudioCapture, CaptureError } from "./audio";
import { copyText, el, h } from "./dom";
import { strings, type Strings, type UILang } from "./i18n";
import { hydrateIcons, icon, type IconName } from "./icons";
import { CodePanel } from "./panels/code";
import { CommandPalette, type Command } from "./palette";
import { PtyPanel } from "./panels/pty";
import { TerminalPanel } from "./panels/terminal";
import * as prefs from "./prefs";
import { SettingsDialog, ShortcutsDialog, type SettingsHost, type UpdateStatus } from "./settings";
import { Sessions } from "./sessions";
import { Shell } from "./shell";
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

import type { Theme } from "./settings";

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

/* ---------------------------------------------------------------- panels */

/**
 * What the side panel can show. Opening one never replaces the conversation:
 * the panel is where the conversation goes to look at something.
 */
type PanelTab = "transcript" | "analysis" | "code" | "terminal";

const PANEL_TABS: readonly PanelTab[] = ["transcript", "analysis", "code", "terminal"];

const PANEL_ICONS: Record<PanelTab, IconName> = {
  transcript: "waveform",
  analysis: "minutes",
  code: "code",
  terminal: "terminal",
};

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

/* ------------------------------------------------------------------ app */

class App {
  private readonly store = new Store({ ...INITIAL, uiLang: detectUILang() });
  private client: StreamClient | null = null;
  private capture: AudioCapture | null = null;
  private prefs: prefs.Prefs = prefs.load();
  private tab: PanelTab = "transcript";
  private codeLabel = "";
  /** The language the labels were last rendered in; see renderLabels. */
  private labelsLang = "";
  private updateState: UpdateStatus | null = null;
  private readonly shell: Shell;
  private readonly sessions: Sessions;
  private terminal: TerminalPanel | null = null;
  private pty: PtyPanel | null = null;
  private termMode: "pty" | "plain" = "pty";
  private ptyAvailable = true;
  private code: CodePanel | null = null;
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
    backend: el<HTMLSpanElement>("status-backend"),
    levelFill: el<HTMLDivElement>("level-fill"),
    transcript: el<HTMLDivElement>("transcript"),
    side: el<HTMLDivElement>("side"),
    stats: el<HTMLElement>("stats"),
    asrLang: el<HTMLSelectElement>("asr-lang"),
    uiLang: el<HTMLElement>("ui-lang"),
    termMode: el<HTMLElement>("term-mode"),
    termModeHint: el<HTMLElement>("term-mode-hint"),
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
    newSession: el<HTMLButtonElement>("new-session"),
    navRecord: el<HTMLButtonElement>("nav-record"),
    navRecordLabel: el<HTMLElement>("nav-record-label"),
    sidebarClose: el<HTMLButtonElement>("sidebar-close"),
    sidebarOpen: el<HTMLButtonElement>("sidebar-open"),
    sessionTitle: el<HTMLElement>("session-title"),
    recPill: el<HTMLButtonElement>("rec-pill"),
    recPillText: el<HTMLElement>("rec-pill-text"),
    panelIcon: el<HTMLElement>("panel-icon"),
    panelTitle: el<HTMLElement>("panel-title"),
    panelClose: el<HTMLButtonElement>("panel-close"),
    cmdk: el<HTMLButtonElement>("cmdk"),
    updatePill: el<HTMLButtonElement>("update-pill"),
    updatePillText: el<HTMLElement>("update-pill-text"),
    updatePillAction: el<HTMLElement>("update-pill-action"),
  };

  private readonly notify = new Notifications(undefined, this.refs.live);
  private readonly settings = new SettingsDialog(
    this.notify,
    strings(detectUILang()),
    this.settingsHost(),
  );
  private readonly shortcuts = new ShortcutsDialog(strings(detectUILang()));
  private readonly palette = new CommandPalette(strings(detectUILang()), () =>
    this.store.get().uiLang,
  );

  constructor() {
    applyTheme(this.theme);
    hydrateIcons(document);
    this.shell = new Shell(el("workbench"), {
      onResize: (region, geometry) => {
        if (region === "sidebar") {
          this.refs.sidebarOpen.hidden = geometry.visible;
          return;
        }
        if (geometry.visible) {
          // xterm and the canvases measure their containers, so a panel that
          // changed width has to be redrawn at the width it now has.
          this.pty?.refit();
          this.redrawCanvases();
        }
        // Undefined only while the shell is still being constructed.
        if (this.shell) this.renderPanelChrome();
      },
    });
    this.sessions = new Sessions(
      el("sessions-host"),
      el("session-list"),
      strings(this.store.get().uiLang),
      this.notify,
      {
        onChange: () => this.renderChrome(this.s),
        onModelClick: () => void this.settings.open("models"),
      },
    );
    this.sessions.create();

    this.bind();
    this.store.subscribe(() => this.render());
    this.render();
    this.levelStrip.render();
    this.timeline.render([], 0);
    void this.loadProviders();
    void this.loadActiveModel();
    void this.loadTerminalCapabilities();
    void this.watchUpdates();

    const redraw = () => this.redrawCanvases();
    window.addEventListener("resize", redraw);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    this.showTab(this.tab);
  }

  private get s(): Strings {
    return strings(this.store.get().uiLang);
  }

  /* ---------------------------------------------------------------- wiring */

  private bind(): void {
    this.refs.toggle.addEventListener("click", () => void this.toggle());
    this.refs.demo.addEventListener("click", () => void this.startDemo());
    this.refs.navRecord.addEventListener("click", () => this.recordFromAnywhere());
    this.refs.recPill.addEventListener("click", () => this.openPanel("transcript"));
    this.refs.newSession.addEventListener("click", () => this.newSession());
    this.refs.sidebarClose.addEventListener("click", () => this.shell.show("sidebar", false));
    this.refs.sidebarOpen.addEventListener("click", () => this.shell.show("sidebar", true));

    // An overlay closes when you press outside it, as a drawer does. Only as an
    // overlay: beside the conversation a region is part of the layout, and a
    // click in the conversation is not a request to put it away.
    el("workbench").addEventListener("pointerdown", (event) => {
      const target = event.target as HTMLElement;
      if (
        this.shell.overlaid("panel") &&
        this.shell.visible("panel") &&
        !el("panel").contains(target) &&
        !target.closest(".head-btn, #rec-pill, #nav-record")
      ) {
        this.shell.show("panel", false);
        this.renderPanelChrome();
      }
      if (
        this.shell.overlaid("sidebar") &&
        this.shell.visible("sidebar") &&
        !el("sidebar").contains(target) &&
        !target.closest("#sidebar-open")
      ) {
        this.shell.show("sidebar", false);
      }
    });
    this.refs.panelClose.addEventListener("click", () => this.togglePanel());
    this.refs.cmdk.addEventListener("click", () => this.palette.show());
    this.refs.updatePill.addEventListener("click", () => void this.restartToUpdate());
    for (const button of document.querySelectorAll<HTMLElement>(".head-btn[data-panel]")) {
      button.addEventListener("click", () => this.togglePanel(button.dataset.panel as PanelTab));
    }

    this.refs.asrLang.addEventListener("change", () => {
      this.store.set({ asrLang: this.refs.asrLang.value as ASRLang });
    });

    this.refs.uiLang.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>("[data-lang]");
      if (target) this.setLanguage(target.dataset.lang as UILang);
    });

    // Left/right on the toggle, which is what a radiogroup promises and what
    // anyone navigating by keyboard will try.
    this.refs.uiLang.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      this.setLanguage(this.store.get().uiLang === "ja" ? "en" : "ja");
      this.refs.uiLang.querySelector<HTMLElement>('[aria-checked="true"]')?.focus();
    });

    this.refs.termMode.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>("[data-mode]");
      if (target) this.setTerminalMode(target.dataset.mode as "pty" | "plain");
    });

    this.refs.themeBtn.addEventListener("click", () => this.cycleTheme());
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

    // A toggle: the same button that opened the filter closes it.
    this.refs.searchToggle.addEventListener("click", () =>
      this.refs.search.hidden ? this.openSearch() : this.closeSearch(),
    );
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
      // A modal owns the keyboard while it is open; a global handler firing
      // behind it would act on a surface the user cannot see.
      if (document.querySelector("dialog[open]")) return;

      const mod = event.ctrlKey || event.metaKey;
      const key = event.key.toLowerCase();

      // The chords come before the typing guard: they are how you get *out*
      // of a text field, and a shortcut that only works with the caret
      // elsewhere is one people stop reaching for.
      if (mod && !event.altKey) {
        if (key === "k") {
          event.preventDefault();
          this.palette.show();
          return;
        }
        if (!event.shiftKey && /^[1-4]$/.test(event.key)) {
          event.preventDefault();
          this.togglePanel(PANEL_TABS[Number(event.key) - 1]);
          return;
        }
        if (key === "b" && !event.shiftKey) {
          event.preventDefault();
          this.shell.toggle("sidebar");
          return;
        }
        if (key === "j" && !event.shiftKey) {
          event.preventDefault();
          this.togglePanel();
          return;
        }
        if (event.key === "`") {
          event.preventDefault();
          this.togglePanel("terminal");
          return;
        }
        if (key === "l" && !event.shiftKey) {
          event.preventDefault();
          this.focusComposer();
          return;
        }
        if (key === "o" && event.shiftKey) {
          event.preventDefault();
          this.newSession();
          return;
        }
        // The physical key first, since Shift+[ is "{" on one layout and
        // something else on the next; the characters as a fallback, because
        // not every input path fills in `code` (synthesized and remapped keys
        // arrive without it, and the shortcut silently did nothing).
        const next = event.code === "BracketRight" || event.key === "]" || event.key === "}";
        const previous = event.code === "BracketLeft" || event.key === "[" || event.key === "{";
        if (event.shiftKey && (next || previous)) {
          event.preventDefault();
          this.sessions.cycle(next ? 1 : -1);
          return;
        }
      }

      const typing = isTyping();
      const inControl = typing || document.activeElement instanceof HTMLButtonElement;

      // Space toggles recording, unless the user is in a control — where space
      // means "activate this button" and stealing it would be hostile.
      if (event.code === "Space" && !inControl && !mod) {
        event.preventDefault();
        this.recordFromAnywhere();
        return;
      }

      if (event.key === "Escape") {
        if (this.store.get().query) this.closeSearch();
        else this.store.set({ selectedQuote: null });
        return;
      }

      if (typing || event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.key === "/") {
        event.preventDefault();
        this.openPanel("transcript");
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

  /**
   * The settings panel's view of this application.
   *
   * Handed over as a small interface rather than `this`, so the dialog can
   * read and write settings and nothing else — it cannot reach into the
   * session, the store, or the render path.
   */
  private settingsHost(): SettingsHost {
    return {
      getPrefs: () => this.prefs,
      setPrefs: (patch) => this.updatePrefs(patch),
      getTheme: () => this.theme,
      setTheme: (theme) => this.applyThemeChoice(theme),
      getUiLang: () => this.store.get().uiLang,
      setUiLang: (uiLang) => this.setLanguage(uiLang),
      getAsrLang: () => this.store.get().asrLang,
      setAsrLang: (asrLang) => {
        this.refs.asrLang.value = asrLang;
        this.store.set({ asrLang });
      },
      onCredentialsChanged: () => void this.loadActiveModel(),
      isRecording: () => {
        const status = this.store.get().status;
        return status === "live" || status === "connecting";
      },
    };
  }

  private updatePrefs(patch: Partial<prefs.Prefs>): void {
    this.prefs = { ...this.prefs, ...patch };
    prefs.save(this.prefs);
    // Gain is the one setting that can take effect without restarting the
    // session, and a volume control that needs a restart is not a volume
    // control. Everything else applies to the next session by construction.
    if (patch.gain !== undefined) this.capture?.setGain(patch.gain);
  }

  private applyThemeChoice(theme: Theme): void {
    this.theme = theme;
    applyTheme(theme);
    // An attribute, not the button's text: the button holds an icon now, and
    // writing a glyph into it replaced the icon with a character.
    this.refs.themeBtn.dataset.theme = theme;
    // The canvases paint with resolved theme colours, so they are stale the
    // instant the palette changes and nothing else redraws them.
    this.levelStrip.render();
    const state = this.store.get();
    this.timeline.render(state.utterances, Math.max(state.durationS, 1));
    // xterm holds resolved colours rather than CSS variables, so it has to be
    // told; everything else in the app repaints from the custom properties.
    this.pty?.setTheme(theme !== "light");
  }

  /** Name why a capture failed, rather than blaming the microphone for all of it. */
  private captureMessage(error: unknown): string {
    const s = this.s;
    if (error instanceof CaptureError) {
      if (error.reason === "no-audio") return s.shareNoAudio;
      if (error.reason === "unsupported") return s.shareUnsupported;
      if (error.reason === "denied") {
        return this.prefs.source === "system" ? s.shareCancelled : s.micDenied;
      }
    }
    return s.micDenied;
  }

  /** Show a tab's contents, whether or not the panel is open. */
  private showTab(tab: PanelTab): void {
    this.tab = tab;
    for (const name of PANEL_TABS) el(`doc-${name}`).hidden = name !== tab;
    this.renderPanelChrome();
  }

  /**
   * Open the side panel on a tab.
   *
   * The code viewer and the terminal are built the first time they are opened,
   * not at startup: a file tree costs a request per directory and a terminal
   * spawns a shell, and neither is owed to someone who came to record.
   */
  private openPanel(tab: PanelTab): void {
    this.showTab(tab);
    this.shell.show("panel", true);
    if (tab === "code") this.ensureCode();
    else if (tab === "terminal") this.openTerminal();
    else if (tab === "transcript") requestAnimationFrame(() => this.redrawCanvases());
    this.renderPanelChrome();
  }

  /** A panel button: open that tab, or close the panel if it is already showing it. */
  private togglePanel(tab?: PanelTab): void {
    if (this.shell.visible("panel") && (!tab || tab === this.tab)) {
      this.shell.show("panel", false);
      this.renderPanelChrome();
      return;
    }
    this.openPanel(tab ?? this.tab);
  }

  private ensureCode(): CodePanel {
    if (!this.code) {
      this.code = new CodePanel(
        { tree: el("code-tree"), view: el("code-view") },
        this.s,
        this.notify,
        {
          onOpen: (label) => {
            this.codeLabel = label;
            this.renderPanelChrome();
          },
        },
      );
      void this.code.activate();
    }
    return this.code;
  }

  /** Start or stop recording from outside the panel, and show what it is doing. */
  private recordFromAnywhere(): void {
    const status = this.store.get().status;
    if (status !== "live" && status !== "connecting") this.openPanel("transcript");
    void this.toggle();
  }

  private newSession(): void {
    this.sessions.create();
    this.sessions.active?.panel.focus();
  }

  /**
   * ⌘L: go to the composer, bringing any text selected in the side panel with
   * you as a quote — the transcript line you were just reading, usually.
   */
  private focusComposer(): void {
    const panel = this.sessions.active?.panel;
    if (!panel) return;
    const selection = window.getSelection();
    const text = selection?.toString().trim() ?? "";
    const anchor = selection?.anchorNode ?? null;
    if (text && anchor && el("panel").contains(anchor)) panel.quote(text);
    else panel.focus();
  }

  private redrawCanvases(): void {
    this.levelStrip.render();
    const state = this.store.get();
    this.timeline.render(state.utterances, Math.max(state.durationS, 1));
  }

  private setLanguage(uiLang: UILang): void {
    if (uiLang !== "ja" && uiLang !== "en") return;
    document.documentElement.lang = uiLang;
    this.settings.setStrings(strings(uiLang));
    this.shortcuts.setStrings(strings(uiLang));
    this.store.set({ uiLang });
  }

  /**
   * Show one of the two terminals.
   *
   * They are different products, not a fallback pair: the interactive one is
   * a real pty behind an emulator and runs `vim`; the plain one is the
   * stripped text a model sees. Someone debugging what the assistant saw
   * wants the second, and it is worth being able to look at exactly that.
   */
  private openTerminal(): void {
    const s = this.s;
    const usePty = this.termMode === "pty" && this.ptyAvailable;

    el("term-pty").hidden = !usePty;
    el("term-plain").hidden = usePty;

    for (const button of this.refs.termMode.querySelectorAll<HTMLElement>("[data-mode]")) {
      const on = button.dataset.mode === this.termMode;
      button.classList.toggle("on", on);
      button.setAttribute("aria-checked", String(on));
    }

    if (usePty) {
      this.pty ??= new PtyPanel(el("term-pty"), s, this.notify);
      // Built on first show, so opening the app does not load a 250 KB
      // emulator for someone who came to record a meeting.
      void this.pty.activate().then(() => this.pty?.refit());
      this.pty.refit();
    } else {
      this.terminal ??= new TerminalPanel(el("term-plain"), s, this.notify);
      this.terminal.focus();
    }
  }

  private setTerminalMode(mode: "pty" | "plain"): void {
    if (mode === "pty" && !this.ptyAvailable) {
      this.notify.info(this.s.terminalNoPty);
      return;
    }
    this.termMode = mode;
    this.openTerminal();
  }

  /** Ask the server which terminals this machine can actually open. */
  private async loadTerminalCapabilities(): Promise<void> {
    try {
      const response = await fetch("/v1/terminal/capabilities");
      if (!response.ok) return;
      const body = (await response.json()) as { pty: boolean };
      this.ptyAvailable = Boolean(body.pty);
      if (!this.ptyAvailable) {
        // Guessing from the platform would be wrong on a Windows box with no
        // pywinpty installed, which is the common case on a fresh machine.
        this.termMode = "plain";
        el<HTMLButtonElement>("term-mode-pty").disabled = true;
        el("term-mode-pty").title = this.s.terminalNoPty;
        this.refs.termModeHint.textContent = this.s.terminalNoPty;
      }
    } catch {
      /* the panel falls back to plain, which always works */
    }
  }

  /**
   * Everything the palette can do.
   *
   * Rebuilt whenever the language changes rather than held: a command's label
   * is part of it, and a stale list is a palette that answers in the language
   * the app is no longer in.
   */
  private buildCommands(): void {
    const panel = (tab: PanelTab) => () => this.openPanel(tab);
    const commands: Command[] = [
      { id: "new-session", ja: "新しいセッション", en: "New session", group: "session", hint: "Ctrl+Shift+O", run: () => this.newSession() },
      { id: "composer", ja: "入力欄へ移動", en: "Focus the composer", group: "session", hint: "Ctrl+L", run: () => this.focusComposer() },
      { id: "next-session", ja: "次のセッション", en: "Next session", group: "session", hint: "Ctrl+Shift+]", run: () => this.sessions.cycle(1) },
      { id: "previous-session", ja: "前のセッション", en: "Previous session", group: "session", hint: "Ctrl+Shift+[", run: () => this.sessions.cycle(-1) },

      { id: "transcript", ja: "文字起こしを表示", en: "Show the transcript", group: "view", hint: "Ctrl+1", run: panel("transcript") },
      { id: "analysis", ja: "議事録を表示", en: "Show the minutes", group: "view", hint: "Ctrl+2", run: panel("analysis") },
      { id: "code", ja: "コードを表示", en: "Show the code", group: "view", hint: "Ctrl+3", run: panel("code") },
      { id: "terminal", ja: "ターミナルを表示", en: "Show the terminal", group: "view", hint: "Ctrl+4", run: panel("terminal") },
      { id: "sidebar", ja: "サイドバーの表示切替", en: "Toggle the sidebar", group: "view", hint: "Ctrl+B", run: () => void this.shell.toggle("sidebar") },
      { id: "panel", ja: "サイドパネルの表示切替", en: "Toggle the side panel", group: "view", hint: "Ctrl+J", run: () => this.togglePanel() },
      { id: "reset-layout", ja: "レイアウトを初期化", en: "Reset layout", group: "view", run: () => { this.shell.reset("sidebar"); this.shell.reset("panel"); } },

      { id: "record", ja: "録音の開始・停止", en: "Start or stop recording", group: "meeting", hint: "Space", run: () => this.recordFromAnywhere() },
      { id: "demo", ja: "デモを再生", en: "Play the demo meeting", group: "meeting", run: () => { this.openPanel("transcript"); void this.startDemo(); } },
      { id: "minutes", ja: "議事録を作成", en: "Generate minutes", group: "meeting", run: () => { this.openPanel("analysis"); this.requestMinutes(); } },
      { id: "copy-transcript", ja: "文字起こしをコピー", en: "Copy the transcript", group: "meeting", run: () => void this.copyTranscript() },
      { id: "copy-minutes", ja: "議事録をコピー", en: "Copy the minutes", group: "meeting", run: () => void this.copyMinutes() },
      { id: "search", ja: "文字起こしを検索", en: "Search the transcript", group: "meeting", hint: "/", run: () => { this.openPanel("transcript"); this.openSearch(); } },

      { id: "settings", ja: "設定を開く", en: "Open settings", group: "app", hint: ",", run: () => void this.settings.open() },
      { id: "plugins", ja: "プラグイン", en: "Plugins", group: "app", run: () => void this.settings.open("plugins") },
      { id: "models", ja: "APIキー", en: "API keys", group: "app", run: () => void this.settings.open("models") },
      { id: "audio", ja: "音声設定", en: "Audio settings", group: "app", run: () => void this.settings.open("audio") },
      { id: "theme", ja: "テーマを切り替え", en: "Toggle theme", group: "app", run: () => this.cycleTheme() },
      { id: "language", ja: "English に切り替え", en: "日本語に切り替え", group: "app", run: () => this.setLanguage(this.store.get().uiLang === "ja" ? "en" : "ja") },
      { id: "shortcuts", ja: "ショートカット一覧", en: "Keyboard shortcuts", group: "app", hint: "?", run: () => this.shortcuts.open() },
    ];
    this.palette.register(commands);
  }

  private cycleTheme(): void {
    const order: Theme[] = ["system", "light", "dark"];
    this.applyThemeChoice(order[(order.indexOf(this.theme) + 1) % order.length]);
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
    // "connecting" counts as busy. Guarding only on "live" left a window
    // between the click and the socket opening in which a second click
    // started a second session.
    const status = this.store.get().status;
    if (status === "live" || status === "connecting") return;
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

  /**
   * Build a client bound to this app's handlers.
   *
   * Closes any previous socket first. Without that, a second session could be
   * opened while the first was still delivering, both would push finals into
   * the same store, and the transcript came out with every line twice — which
   * is exactly what happened when the demo button was pressed twice before the
   * first connection had finished opening.
   */
  private connect(): StreamClient {
    this.client?.close();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const client = new StreamClient(`${scheme}://${location.host}/v1/stream`, {
      onMessage: (message) => {
        // A late frame from a socket we have already replaced belongs to a
        // session the user ended; applying it would revive dead state.
        if (this.client !== client) return;
        this.onMessage(message);
      },
      onError: () => {
        if (this.client === client) this.fail(this.s.cannotConnect);
      },
      onClose: () => {
        if (this.client !== client) return;
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

    client.start({
      language: this.store.get().asrLang,
      partialIntervalMs: this.prefs.partialIntervalMs,
      // Omitted when the user wants the language defaults, because the server
      // keeps a longer silence window for Japanese and sending a number here
      // would silently overwrite it.
      ...(this.prefs.useEndpointDefaults
        ? {}
        : {
            silenceToEndMs: this.prefs.silenceToEndMs,
            speechThresholdDb: this.prefs.speechThresholdDb,
          }),
    });

    this.capture = new AudioCapture({
      source: this.prefs.source,
      deviceId: this.prefs.inputDeviceId,
      gain: this.prefs.gain,
      echoCancellation: this.prefs.echoCancellation,
      noiseSuppression: this.prefs.noiseSuppression,
      autoGainControl: this.prefs.autoGainControl,
      onFrame: (frame) => client.sendAudio(frame),
      onLevel: (level) => {
        this.refs.levelFill.style.width = `${Math.min(100, level * 140)}%`;
        this.levelStrip.push(level);
      },
      // Pressing "Stop sharing" in the browser's own bar ends the track and
      // says nothing else. Without this the session sits recording silence.
      onSourceEnded: () => {
        this.notify.info(this.s.sourceEnded);
        void this.stop();
      },
    });

    try {
      await this.capture.start();
    } catch (error) {
      this.fail(this.captureMessage(error));
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

    this.refs.asrLang.value = state.asrLang;
    this.refs.themeBtn.title = s.theme;
    this.refs.themeBtn.setAttribute("aria-label", s.theme);

    this.renderLabels(s);
    this.renderTranscript(s);
    this.renderSide(s);
    this.renderStats(s);
    this.renderChrome(s);
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

  /**
   * Everything whose words depend on the language.
   *
   * Guarded on the language actually changing. This runs from every store
   * update, and it used to rebuild the palette's whole command list and
   * re-label every panel several times a second while recording — the most
   * expensive thing the page did, to change nothing.
   */
  private renderLabels(s: Strings): void {
    const lang = this.store.get().uiLang;
    if (lang === this.labelsLang) return;
    this.labelsLang = lang;

    el("t-level").textContent = s.level;
    el("t-speakers").textContent = s.speakers;
    el("tab-minutes").textContent = s.minutes;
    el("tab-routing").textContent = s.routing;
    el("tab-metrics").textContent = s.metrics;
    el("t-asr-lang").textContent = s.asrLanguage;
    el("t-new-session").textContent = s.newSession;
    el("t-sessions").textContent = s.sessionsHeading;
    // The language toggle carries its own bilingual aria-label.
    this.refs.uiLang.setAttribute("aria-label", `${s.language} / Language`);

    for (const button of [this.refs.sidebarClose, this.refs.sidebarOpen]) {
      button.title = `${s.sidebarToggle} (Ctrl+B)`;
      button.setAttribute("aria-label", s.sidebarToggle);
    }
    this.refs.newSession.title = `${s.newSession} (Ctrl+Shift+O)`;
    this.refs.navRecord.title = `${s.recordMeeting} (Space)`;
    this.refs.panelClose.title = `${s.panelClose} (Ctrl+J)`;
    this.refs.panelClose.setAttribute("aria-label", s.panelClose);
    this.refs.cmdk.setAttribute("aria-label", s.paletteSearch);

    this.palette.setStrings(s);
    this.buildCommands();
    this.sessions.setStrings(s);
    this.terminal?.setStrings(s);
    this.pty?.setStrings(s);
    this.code?.setStrings(s);

    for (const button of this.refs.uiLang.querySelectorAll<HTMLElement>("[data-lang]")) {
      const on = button.dataset.lang === lang;
      button.classList.toggle("on", on);
      button.setAttribute("aria-checked", String(on));
      // Roving tabindex: one stop for the pair, arrows move within it.
      (button as HTMLButtonElement).tabIndex = on ? 0 : -1;
    }
    el("term-mode-pty").textContent = s.terminalInteractive;
    el("term-mode-plain").textContent = s.terminalPlain;
    this.refs.termModeHint.textContent = this.ptyAvailable ? s.terminalModeHint : s.terminalNoPty;
    this.refs.termModeHint.title = this.refs.termModeHint.textContent;

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
    this.renderPanelChrome();
    this.renderUpdatePill();
  }

  /**
   * The sidebar's "Update ready" button, in the installed app only.
   *
   * Polled from the local server rather than pushed: one small loopback
   * request a minute, for a state that changes a few times a week. A push
   * channel would be machinery for its own sake.
   */
  private async watchUpdates(): Promise<void> {
    const refresh = async (): Promise<boolean> => {
      try {
        const response = await fetch("/v1/update");
        // Not the installed app: there is nothing to watch, so stop asking.
        if (!response.ok) return false;
        this.updateState = (await response.json()) as UpdateStatus;
      } catch {
        return true; // transient; the next poll will tell
      }
      this.renderUpdatePill();
      return true;
    };
    if (!(await refresh())) return;
    window.setInterval(() => void refresh(), 60_000);
    window.addEventListener("focus", () => void refresh());
  }

  private renderUpdatePill(): void {
    const update = this.updateState;
    const ready = update?.state === "ready";
    this.refs.updatePill.hidden = !ready;
    if (!ready || !update) return;
    const s = this.s;
    this.refs.updatePillText.textContent = `${s.updatePill} · ${update.available}`;
    this.refs.updatePillAction.textContent = s.updatePillAction;
    this.refs.updatePill.title = s.updateRestart;
  }

  private async restartToUpdate(): Promise<void> {
    const status = this.store.get().status;
    // Restarting ends the process, and a recording in progress would go with it.
    if (status === "live" || status === "connecting") {
      this.notify.error(this.s.updateStopRecording);
      return;
    }
    try {
      const response = await fetch("/v1/update/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ relaunch: true }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
        this.notify.error(String(body.detail ?? response.status));
        return;
      }
      this.refs.updatePill.disabled = true;
      this.notify.info(this.s.updateApplying);
    } catch {
      this.notify.error(this.s.cannotConnect);
    }
  }

  private renderTranscript(s: Strings): void {
    const state = this.store.get();
    const { utterances, committed, pending, selectedQuote, query } = state;
    const host = this.refs.transcript;
    // Rebuilt on every update, so where the reader was has to be carried
    // across the rebuild: follow the tail only if they were already at it.
    const followTail = host.scrollHeight - host.scrollTop - host.clientHeight < 80;
    const previousTop = host.scrollTop;
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
      row.dataset.index = String(index);
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

    // Scrolled within the transcript itself. scrollIntoView also scrolls every
    // scrollable ancestor, which nudges a fixed-height frame out of place, and
    // following the tail unconditionally yanked a reader who had scrolled up
    // back to the bottom on every new line.
    const target = searching
      ? host.querySelector<HTMLElement>(".utt:not(.dimmed)")
      : host.querySelector<HTMLElement>("#cited-utterance");
    if (target) {
      const offset = target.getBoundingClientRect().top - host.getBoundingClientRect().top;
      host.scrollTop += offset - host.clientHeight / 2 + target.clientHeight / 2;
    } else {
      host.scrollTop = followTail ? host.scrollHeight : previousTop;
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

    if (!selected || !claim.quote) return node;

    // The source, shown under the claim itself. The transcript is another tab
    // now, and a highlight in a tab you cannot see is a citation nobody checks.
    const source = h("div", { class: "claim-source" });
    const citation = findCitation(state.utterances, claim.quote);
    if (citation) {
      const utterance = state.utterances[citation.index];
      const speaker = utterance.speaker || "—";
      const who = h("span", { class: "claim-source-who", text: `${speaker} · ${fmtDuration(utterance.start)}` });
      who.style.setProperty("--spk", `var(--spk-${speakerColorIndex(speaker)})`);
      const said = h("p", { class: "claim-source-text ja" });
      if (citation.raw) {
        const at = utterance.text.indexOf(citation.raw);
        said.append(
          utterance.text.slice(0, at),
          h("mark", { class: "hit", text: citation.raw }),
          utterance.text.slice(at + citation.raw.length),
        );
      } else {
        said.textContent = utterance.text;
      }
      const open = h("button", { class: "btn ghost sm", type: "button", text: s.showInTranscript });
      open.addEventListener("click", () => this.openPanel("transcript"));
      source.append(who, said, open);
    } else {
      source.append(h("p", { class: "claim-source-text ja", text: `「${claim.quote}」` }));
    }
    return h("div", { class: "claim-wrap" }, node, source);
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

  /** The frame around the conversation: its title, and whether a meeting is recording. */
  private renderChrome(s: Strings): void {
    const state = this.store.get();
    const recording = state.status === "live" || state.status === "connecting";
    const title = this.sessions.active?.title ?? "";
    this.refs.sessionTitle.textContent = title || s.sessionUntitled;
    const pageTitle = title ? `${title} — koe` : "koe 声 — bilingual voice AI";
    if (document.title !== pageTitle) document.title = pageTitle;

    const clock = fmtDuration(state.durationS);
    this.refs.recPill.hidden = !recording;
    this.refs.recPillText.textContent = `${s.recordingLive} ${clock}`;
    this.refs.navRecordLabel.textContent = recording ? `${s.stop} · ${clock}` : s.recordMeeting;
    this.refs.navRecord.classList.toggle("live", recording);
    this.sessions.setModel(state.activeLlm);
  }

  /** Which panel button is lit, and what the panel header says. */
  private renderPanelChrome(): void {
    const s = this.s;
    const open = this.shell.visible("panel");
    const labels: Record<PanelTab, string> = {
      transcript: s.transcript,
      analysis: s.minutes,
      code: s.wsCode,
      terminal: s.wsTerminal,
    };
    PANEL_TABS.forEach((tab, index) => {
      const button = document.querySelector<HTMLElement>(`.head-btn[data-panel="${tab}"]`);
      if (!button) return;
      const on = open && tab === this.tab;
      button.classList.toggle("on", on);
      button.setAttribute("aria-pressed", String(on));
      button.title = `${labels[tab]} (Ctrl+${index + 1})`;
      const label = button.querySelector(".label");
      if (label) label.textContent = labels[tab];
    });
    this.refs.panelTitle.textContent =
      this.tab === "code" && this.codeLabel ? basename(this.codeLabel) : labels[this.tab];
    if (this.refs.panelIcon.dataset.tab !== this.tab) {
      this.refs.panelIcon.dataset.tab = this.tab;
      this.refs.panelIcon.replaceChildren(icon(PANEL_ICONS[this.tab], 16));
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
