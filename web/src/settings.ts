/**
 * The settings panel, and the keyboard help.
 *
 * Built on the native `<dialog>` element rather than a positioned div, so the
 * browser supplies the focus trap, Escape-to-close, inertness of the page
 * behind it, and the right semantics for assistive technology. Those are the
 * four things hand-rolled modals reliably get wrong.
 *
 * Five sections, in the order someone actually needs them: **Audio** first,
 * because a meeting tool that is listening to the wrong thing is useless no
 * matter how good the rest is; then **Recognition**, **Models**, and the two
 * that are matters of taste.
 *
 * Two rules run through the whole panel.
 *
 * **Every control changes something real.** The endpointing sliders are sent
 * on the next session and clamped server-side; the gain slider is applied to a
 * live `GainNode` and takes effect mid-recording. There is no setting here
 * that is only remembered.
 *
 * **Nothing claims more than it knows.** The audio-source options say what
 * Windows will and will not give you rather than letting someone pick a single
 * application window and quietly transcribe silence, and a key is never
 * reported as valid until something has checked it.
 *
 * The key-entry flow follows three of its own:
 *
 * **A key is write-only.** The input is always empty on open. There is nothing
 * to read back — the server returns a fingerprint and never the key.
 *
 * **Saving and testing are separate.** Verification costs a request, and a
 * save that silently spends money is a surprise.
 *
 * **The failure mode is named.** "Invalid" and "error" are different problems:
 * one means fix your key, the other means check your network.
 */

import {
  AudioCapture,
  CaptureError,
  canCaptureSystemAudio,
  listInputDevices,
  type InputDevice,
} from "./audio";
import { h } from "./dom";
import type { Strings, UILang } from "./i18n";
import { DEFAULTS, LIMITS, type AudioSource, type Prefs } from "./prefs";
import type { ASRLang } from "./state";
import type { Notifications } from "./toast";

export type Theme = "light" | "dark" | "system";

/**
 * What the panel needs from the application.
 *
 * An interface rather than a reference to `App` so the dialog cannot reach
 * into rendering or session state: it reads and writes settings, and that is
 * the whole of its access.
 */
export interface SettingsHost {
  getPrefs(): Prefs;
  setPrefs(patch: Partial<Prefs>): void;
  getTheme(): Theme;
  setTheme(theme: Theme): void;
  getUiLang(): UILang;
  setUiLang(lang: UILang): void;
  getAsrLang(): ASRLang;
  setAsrLang(lang: ASRLang): void;
  /** A key was added or removed; the live provider may have changed. */
  onCredentialsChanged(): void;
  /** True while a session owns the audio device. */
  isRecording(): boolean;
}

export interface ProviderCredential {
  provider: string;
  label: string;
  modality: string;
  docs_url: string;
  env_var: string;
  configured: boolean;
  source: "environment" | "store" | "none";
  fingerprint: string;
  status: "unknown" | "valid" | "invalid" | "error";
  verified_at: number | null;
  detail: string;
  /** Enumerable form of `detail`; empty when the outcome is a raw error. */
  code: string;
  editable: boolean;
  /** False when this build lacks the vendor SDK, whatever the key says. */
  available: boolean;
}

/**
 * A rejection from the server.
 *
 * `detail` is English and always present; `code` is what lets this client say
 * the same thing in the language the reader chose. Matching on English prose
 * would break the moment the server reworded a sentence.
 */
interface ServerError {
  detail?: string;
  code?: string;
  context?: Record<string, string>;
}

interface CredentialListing {
  providers: ProviderCredential[];
  active_llm: string;
  forced_mock: boolean;
}

interface Health {
  version: string;
  japanese_tokenizer: string;
}

export type Section =
  | "audio"
  | "recognition"
  | "models"
  | "plugins"
  | "local"
  | "vocabulary"
  | "appearance"
  | "about";

const SECTIONS: readonly Section[] = [
  "audio",
  "recognition",
  "models",
  "plugins",
  "local",
  "vocabulary",
  "appearance",
  "about",
];

interface WhisperSize {
  readonly id: string;
  readonly label: string;
  readonly download_mb: number;
  readonly typical_rtf: number;
  readonly suitable_ja: boolean;
  readonly suitable_en: boolean;
}

interface DiscoveredServer {
  readonly server_id: string;
  readonly label: string;
  readonly base_url: string;
  readonly ready: boolean;
  readonly local: boolean;
  readonly note: string;
  readonly models: readonly { id: string; parameters: string; quantization: string }[];
}

interface LocalState {
  readonly llm: {
    readonly active: boolean;
    readonly in_use: boolean;
    readonly reason: string;
    readonly base_url: string;
    readonly model: string;
    readonly prefer: boolean;
  };
  readonly asr: {
    readonly active: boolean;
    readonly enabled: boolean;
    readonly installed: boolean;
    readonly reason: string;
    readonly model: string;
    readonly device: string;
    readonly sizes: readonly WhisperSize[];
  };
  readonly known_servers: readonly {
    id: string;
    label: string;
    port: number;
    docs_url: string;
  }[];
}

interface VocabularyState {
  readonly enabled: boolean;
  readonly text: string;
  readonly entries: number;
  readonly terms: readonly string[];
}

interface PluginRecord {
  name: string;
  origin: string;
  description: string;
  version: string;
  builtin: boolean;
  enabled: boolean;
  active: boolean;
  error: string;
}

interface PluginListing {
  plugins: PluginRecord[];
  directory: string;
}

interface ToolRecord {
  name: string;
  source: string;
  dangerous: boolean;
}

export class SettingsDialog {
  private readonly dialog: HTMLDialogElement;
  private readonly nav: HTMLElement;
  private readonly body: HTMLElement;
  private section: Section = "audio";
  private listing: CredentialListing | null = null;
  private health: Health | null = null;
  private devices: InputDevice[] = [];
  private pluginListing: PluginListing | null = null;
  private local: LocalState | null = null;
  private discovered: DiscoveredServer[] = [];
  private scanning = false;
  private scanned = false;
  private vocabulary: VocabularyState | null = null;
  private vocabularyDraft: string | null = null;
  private toolRecords: ToolRecord[] = [];
  /** A short-lived capture used only to prove the chosen source works. */
  private probe: AudioCapture | null = null;
  private probeTimer = 0;

  constructor(
    private readonly notify: Notifications,
    private strings: Strings,
    private readonly host: SettingsHost,
  ) {
    this.dialog = h("dialog", { class: "modal wide", "aria-labelledby": "settings-title" });
    this.nav = h("nav", { class: "modal-nav", "aria-label": "sections" });
    this.body = h("div", { class: "modal-body" });
    this.build();
    document.body.append(this.dialog);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    const title = this.dialog.querySelector("#settings-title");
    if (title) title.textContent = strings.settings;
    const close = this.dialog.querySelector<HTMLButtonElement>(".modal-foot .btn");
    if (close) close.textContent = strings.close;
    if (this.dialog.open) this.render();
  }

  private build(): void {
    const head = h(
      "div",
      { class: "modal-head" },
      h("h2", { class: "modal-title", id: "settings-title", text: this.strings.settings }),
    );

    const foot = h("div", { class: "modal-foot" });
    const close = h("button", { class: "btn", type: "button", text: this.strings.close });
    close.addEventListener("click", () => this.dialog.close());
    foot.append(h("div", { style: "flex:1" }), close);

    // Arrow keys move between sections, which is what a vertical list of
    // buttons acting as a tablist has to support to be usable by keyboard.
    this.nav.addEventListener("keydown", (event) => {
      const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
      if (step === 0) return;
      event.preventDefault();
      const at = SECTIONS.indexOf(this.section);
      this.show(SECTIONS[(at + step + SECTIONS.length) % SECTIONS.length]);
      this.nav.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus();
    });

    // Escape and the backdrop close a <dialog> without going through the
    // Close button, so releasing the microphone hangs off the close event
    // rather than off the handler for one of the ways to trigger it.
    this.dialog.addEventListener("close", () => void this.stopProbe());

    this.dialog.append(head, h("div", { class: "modal-split" }, this.nav, this.body), foot);
  }

  async open(section?: Section): Promise<void> {
    // Opening straight to a section is what makes "API keys" a command rather
    // than an instruction to open settings and then find the third tab.
    if (section && SECTIONS.includes(section)) this.section = section;
    this.dialog.showModal();
    this.render();
    // Fetched in parallel and rendered as they land, so the panel is usable
    // immediately rather than blank until the slowest request returns.
    await Promise.all([
      this.loadCredentials(),
      this.loadHealth(),
      this.loadDevices(),
      this.loadPlugins(),
      this.loadVocabulary(),
      this.loadLocal(),
    ]);
    this.render();
  }

  private show(section: Section): void {
    this.section = section;
    this.render();
  }

  /* ------------------------------------------------------------- loading */

  private async loadCredentials(): Promise<void> {
    try {
      const response = await fetch("/v1/credentials");
      if (response.ok) this.listing = (await response.json()) as CredentialListing;
    } catch {
      /* the panel renders without it; the Models section says so */
    }
  }

  private async loadHealth(): Promise<void> {
    try {
      const response = await fetch("/health");
      if (response.ok) this.health = (await response.json()) as Health;
    } catch {
      /* About degrades to showing nothing rather than failing */
    }
  }

  private async loadDevices(): Promise<void> {
    this.devices = await listInputDevices();
  }

  private async loadVocabulary(): Promise<void> {
    try {
      const response = await fetch("/v1/vocabulary");
      if (response.ok) {
        this.vocabulary = (await response.json()) as VocabularyState;
        // Only adopt the server's text when nothing is being typed. Clobbering
        // a half-written list because a background refresh landed is the kind
        // of thing people do not report, they just stop using the feature.
        if (this.vocabularyDraft === null) this.vocabularyDraft = this.vocabulary.text;
      }
    } catch {
      /* the section says it could not load rather than rendering empty */
    }
  }

  private async loadPlugins(): Promise<void> {
    try {
      const [plugins, tools] = await Promise.all([
        fetch("/v1/plugins").then((r) => (r.ok ? r.json() : null)),
        fetch("/v1/tools").then((r) => (r.ok ? r.json() : null)),
      ]);
      if (plugins) this.pluginListing = plugins as PluginListing;
      if (tools) this.toolRecords = (tools as { tools: ToolRecord[] }).tools;
    } catch {
      /* the section says it could not load rather than rendering empty */
    }
  }

  /* -------------------------------------------------------------- render */

  private render(): void {
    const s = this.strings;
    const labels: Record<Section, string> = {
      audio: s.secAudio,
      recognition: s.secRecognition,
      models: s.secModels,
      plugins: s.secPlugins,
      local: s.secLocal,
      vocabulary: s.secVocabulary,
      appearance: s.secAppearance,
      about: s.secAbout,
    };

    this.nav.replaceChildren(
      ...SECTIONS.map((section) => {
        const selected = section === this.section;
        const button = h("button", {
          class: `nav-item${selected ? " on" : ""}`,
          type: "button",
          role: "tab",
          "aria-selected": String(selected),
          text: labels[section],
        });
        button.tabIndex = selected ? 0 : -1;
        button.addEventListener("click", () => this.show(section));
        return button;
      }),
    );

    this.body.replaceChildren(
      ...(this.section === "audio"
        ? this.audioSection()
        : this.section === "recognition"
          ? this.recognitionSection()
          : this.section === "models"
            ? this.modelsSection()
            : this.section === "plugins"
              ? this.pluginsSection()
              : this.section === "local"
                ? this.localSection()
                : this.section === "vocabulary"
                ? this.vocabularySection()
                : this.section === "appearance"
                  ? this.appearanceSection()
                  : this.aboutSection()),
    );
    this.body.scrollTop = 0;
  }

  /* --------------------------------------------------------------- audio */

  private audioSection(): HTMLElement[] {
    const s = this.strings;
    const prefs = this.host.getPrefs();
    const parts: HTMLElement[] = [];

    const hints: Record<AudioSource, string> = {
      microphone: s.sourceMicHint,
      system: s.sourceSystemHint,
      both: s.sourceBothHint,
    };
    const sourceLabels: Record<AudioSource, string> = {
      microphone: s.sourceMic,
      system: s.sourceSystem,
      both: s.sourceBoth,
    };

    parts.push(this.heading(s.audioSource));
    const group = h("div", { class: "choices", role: "radiogroup", "aria-label": s.audioSource });
    const sources: AudioSource[] = canCaptureSystemAudio()
      ? ["microphone", "system", "both"]
      : ["microphone"];

    for (const source of sources) {
      const on = prefs.source === source;
      const choice = h("button", {
        class: `choice${on ? " on" : ""}`,
        type: "button",
        role: "radio",
        "aria-checked": String(on),
      });
      choice.append(
        h("span", { class: "choice-name", text: sourceLabels[source] }),
        h("span", { class: "choice-hint", text: hints[source] }),
      );
      choice.addEventListener("click", () => {
        this.host.setPrefs({ source });
        this.render();
      });
      group.append(choice);
    }
    parts.push(group);

    if (prefs.source !== "microphone") {
      parts.push(h("p", { class: "note callout", text: s.windowsAudioNote }));
    }

    if (prefs.source !== "system") {
      parts.push(this.heading(s.inputDevice));
      const row = h("div", { class: "row" });
      const select = h("select", { class: "select grow" }) as HTMLSelectElement;
      select.append(h("option", { value: "", text: s.systemDefault }));
      for (const device of this.devices) {
        select.append(h("option", { value: device.deviceId, text: device.label }));
      }
      select.value = this.devices.some((d) => d.deviceId === prefs.inputDeviceId)
        ? prefs.inputDeviceId
        : "";
      select.addEventListener("change", () => {
        this.host.setPrefs({ inputDeviceId: select.value });
      });

      const rescan = h("button", { class: "btn ghost", type: "button", text: s.refreshDevices });
      rescan.addEventListener("click", () => {
        void this.loadDevices().then(() => this.render());
      });
      row.append(select, rescan);
      parts.push(row);

      if (this.devices.length === 0) {
        parts.push(h("p", { class: "note", text: s.noInputDevices }));
      } else if (this.devices.every((d) => /^Microphone \d+$/.test(d.label))) {
        // Browsers withhold device labels until the page holds a media
        // permission, so a list of "Microphone 1/2/3" is expected, not broken.
        parts.push(h("p", { class: "note", text: s.deviceNamesHidden }));
      }

      parts.push(this.heading(s.processing));
      parts.push(
        this.toggle(s.echoCancellation, prefs.echoCancellation, (on) =>
          this.host.setPrefs({ echoCancellation: on }),
        ),
        this.toggle(s.noiseSuppression, prefs.noiseSuppression, (on) =>
          this.host.setPrefs({ noiseSuppression: on }),
        ),
        this.toggle(s.autoGainControl, prefs.autoGainControl, (on) =>
          this.host.setPrefs({ autoGainControl: on }),
        ),
        h("p", { class: "note", text: s.processingHint }),
      );
    }

    parts.push(this.heading(s.inputGain));
    parts.push(
      this.slider(
        prefs.gain,
        LIMITS.gain,
        (value) => `${value.toFixed(2)}×`,
        (value) => {
          this.host.setPrefs({ gain: value });
          // The probe is a live capture, so the slider moves its meter too —
          // which is the whole point of being able to hear yourself.
          this.probe?.setGain(value);
        },
      ),
      h("p", { class: "note", text: s.inputGainHint }),
    );

    parts.push(this.heading(s.testInput));
    parts.push(this.inputTest(), h("p", { class: "note", text: s.testInputHint }));

    return parts;
  }

  /**
   * A live level meter for the chosen source.
   *
   * A device picker with no meter tells you which microphone you *selected*,
   * not which one is working — and the difference between those two shows up
   * after the meeting, in a transcript of silence. This opens the real capture
   * path with the real settings, so what it proves is what will happen when
   * recording starts.
   */
  private inputTest(): HTMLElement {
    const s = this.strings;
    const row = h("div", { class: "slider-row" });

    const meter = h("div", { class: "level", role: "meter", "aria-label": s.testInput });
    const fill = h("div", { class: "level-fill" });
    meter.append(fill);

    const button = h("button", {
      class: "btn ghost",
      type: "button",
      text: this.probe ? s.stopTest : s.testInput,
    }) as HTMLButtonElement;

    button.addEventListener("click", () => {
      if (this.probe) {
        void this.stopProbe();
        this.render();
        return;
      }
      if (this.host.isRecording()) {
        // One capture at a time: taking the device now would fight the session
        // that is already using it.
        this.notify.error(s.testWhileRecording);
        return;
      }
      void this.startProbe(fill, button);
    });

    row.append(meter, button);
    return row;
  }

  private async startProbe(fill: HTMLElement, button: HTMLButtonElement): Promise<void> {
    const s = this.strings;
    const prefs = this.host.getPrefs();

    // A permission prompt — or the screen picker — can sit there for as long
    // as the user takes to answer. Without this, every click during that wait
    // opens another capture.
    button.disabled = true;
    const probe = new AudioCapture({
      source: prefs.source,
      deviceId: prefs.inputDeviceId,
      gain: prefs.gain,
      echoCancellation: prefs.echoCancellation,
      noiseSuppression: prefs.noiseSuppression,
      autoGainControl: prefs.autoGainControl,
      // Frames are discarded: nothing is transcribed, nothing is sent, and
      // nothing is billed. Only the level is wanted.
      onFrame: () => {},
      onLevel: (level) => {
        fill.style.width = `${Math.min(100, level * 140)}%`;
      },
      onSourceEnded: () => void this.stopProbe(),
    });

    try {
      await probe.start();
    } catch (error) {
      button.disabled = false;
      this.notify.error(
        error instanceof CaptureError && error.reason === "no-audio"
          ? s.shareNoAudio
          : error instanceof CaptureError && error.reason === "denied"
            ? prefs.source === "system"
              ? s.shareCancelled
              : s.micDenied
            : s.micDenied,
      );
      return;
    }

    this.probe = probe;
    button.disabled = false;
    button.textContent = s.stopTest;
    // Bounded, so a panel left open does not hold the microphone — and the
    // recording indicator — for the rest of the day.
    this.probeTimer = window.setTimeout(() => {
      void this.stopProbe();
      this.render();
    }, 20_000);
  }

  private async stopProbe(): Promise<void> {
    window.clearTimeout(this.probeTimer);
    const probe = this.probe;
    this.probe = null;
    await probe?.stop();
  }

  /* --------------------------------------------------------- recognition */

  private recognitionSection(): HTMLElement[] {
    const s = this.strings;
    const prefs = this.host.getPrefs();
    const parts: HTMLElement[] = [];

    parts.push(this.heading(s.asrLanguage));
    const language = h("select", { class: "select grow" }) as HTMLSelectElement;
    language.append(
      h("option", { value: "ja", text: "日本語" }),
      h("option", { value: "en", text: "English" }),
      h("option", { value: "unknown", text: s.auto }),
    );
    language.value = this.host.getAsrLang();
    language.addEventListener("change", () => {
      this.host.setAsrLang(language.value as ASRLang);
      this.render();
    });
    parts.push(h("div", { class: "row" }, language));

    parts.push(this.heading(s.partialInterval));
    parts.push(
      this.slider(
        prefs.partialIntervalMs,
        LIMITS.partialIntervalMs,
        (value) => `${value.toFixed(0)} ms`,
        (value) => this.host.setPrefs({ partialIntervalMs: value }),
      ),
      h("p", { class: "note", text: s.partialIntervalHint }),
    );

    parts.push(this.heading(s.endpointing));
    parts.push(
      this.toggle(s.useLanguageDefaults, prefs.useEndpointDefaults, (on) => {
        this.host.setPrefs({ useEndpointDefaults: on });
        this.render();
      }),
      h("p", { class: "note", text: s.useLanguageDefaultsHint }),
    );

    if (!prefs.useEndpointDefaults) {
      parts.push(
        h("p", { class: "label", text: s.silenceToEnd }),
        this.slider(
          prefs.silenceToEndMs,
          LIMITS.silenceToEndMs,
          (value) => `${value.toFixed(0)} ms`,
          (value) => this.host.setPrefs({ silenceToEndMs: value }),
        ),
        h("p", { class: "note", text: s.silenceToEndHint }),
        h("p", { class: "label", text: s.speechThreshold }),
        this.slider(
          prefs.speechThresholdDb,
          LIMITS.speechThresholdDb,
          (value) => `${value.toFixed(0)} dB`,
          (value) => this.host.setPrefs({ speechThresholdDb: value }),
        ),
        h("p", { class: "note", text: s.speechThresholdHint }),
      );
    }

    return parts;
  }

  /* -------------------------------------------------------------- models */

  private modelsSection(): HTMLElement[] {
    const s = this.strings;
    const parts: HTMLElement[] = [];
    const listing = this.listing;

    if (!listing) {
      parts.push(h("p", { class: "note", text: s.cannotConnect }));
      return parts;
    }

    // What is actually in use right now, stated before any configuration —
    // it is the question someone opening this panel is asking.
    const usingMock = !listing.active_llm || listing.active_llm.startsWith("mock");
    const active = h("div", { class: "row", style: "margin-bottom:14px" });
    active.append(
      h("span", { class: "note", text: `${s.activeModel}:` }),
      h("span", {
        class: usingMock ? "chip warn" : "chip live",
        text: usingMock ? s.usingMock : listing.active_llm,
      }),
    );
    parts.push(active);
    if (usingMock) {
      parts.push(h("p", { class: "note", style: "margin:-8px 0 14px", text: s.usingMockHint }));
    }

    parts.push(h("p", { class: "note", style: "margin:0 0 14px", text: s.apiKeysHint }));
    for (const provider of listing.providers) parts.push(this.providerCard(provider));
    parts.push(
      h("p", {
        class: "note",
        style: "margin-top:18px;padding-top:14px;border-top:1px solid var(--border)",
        text: s.storageNote,
      }),
    );
    return parts;
  }

  /* ---------------------------------------------------------- appearance */

  private appearanceSection(): HTMLElement[] {
    const s = this.strings;
    const parts: HTMLElement[] = [];

    parts.push(this.heading(s.themeLabel));
    const themes: Array<[Theme, string]> = [
      ["system", s.themeSystem],
      ["light", s.themeLight],
      ["dark", s.themeDark],
    ];
    const current = this.host.getTheme();
    const group = h("div", { class: "segmented", role: "radiogroup", "aria-label": s.themeLabel });
    for (const [theme, label] of themes) {
      const on = theme === current;
      const button = h("button", {
        class: `seg${on ? " on" : ""}`,
        type: "button",
        role: "radio",
        "aria-checked": String(on),
        text: label,
      });
      button.addEventListener("click", () => {
        this.host.setTheme(theme);
        this.render();
      });
      group.append(button);
    }
    parts.push(group);

    parts.push(this.heading(s.uiLanguage));
    const language = h("select", { class: "select grow" }) as HTMLSelectElement;
    language.append(
      h("option", { value: "ja", text: "日本語" }),
      h("option", { value: "en", text: "English" }),
    );
    language.value = this.host.getUiLang();
    language.addEventListener("change", () => {
      this.host.setUiLang(language.value as UILang);
    });
    parts.push(h("div", { class: "row" }, language));
    return parts;
  }

  /* --------------------------------------------------------------- about */

  private aboutSection(): HTMLElement[] {
    const s = this.strings;
    const parts: HTMLElement[] = [];
    const rows: Array<[string, string]> = [];

    if (this.health) {
      rows.push([s.version, this.health.version]);
      rows.push([s.japaneseTokenizer, this.health.japanese_tokenizer]);
    }
    if (this.listing) {
      const mock = !this.listing.active_llm || this.listing.active_llm.startsWith("mock");
      rows.push([s.activeModel, mock ? s.usingMock : this.listing.active_llm]);
    }

    const table = h("table", { class: "tbl" });
    const body = h("tbody");
    for (const [label, value] of rows) {
      body.append(h("tr", {}, h("td", { text: label }), h("td", { class: "num", text: value })));
    }
    table.append(body);
    parts.push(table);

    parts.push(this.heading(s.restoreDefaults));
    const reset = h("button", { class: "btn ghost", type: "button", text: s.restoreDefaults });
    reset.addEventListener("click", () => {
      this.host.setPrefs({ ...DEFAULTS });
      this.notify.ok(s.restored);
      this.render();
    });
    parts.push(h("div", { class: "row" }, reset), h("p", { class: "note", text: s.restoreDefaultsHint }));
    return parts;
  }

  /* ------------------------------------------------------------- plugins */

  /* --------------------------------------------------------------- local */

  private localSection(): HTMLElement[] {
    const s = this.strings;
    const state = this.local;
    const parts: HTMLElement[] = [];

    if (!state) {
      parts.push(h("p", { class: "note", text: s.cannotConnect }));
      return parts;
    }

    parts.push(h("p", { class: "note", style: "margin:0 0 16px", text: s.localHint }));

    // ---- language model -------------------------------------------------
    parts.push(h("p", { class: "label", text: s.localLlm }));

    const status = h("div", { class: "row", style: "margin:6px 0 10px;gap:8px" });
    if (state.llm.in_use) {
      status.append(h("span", { class: "chip live", text: s.localInUse }));
    } else if (state.llm.active) {
      status.append(h("span", { class: "chip", text: s.localReady }));
    }
    status.append(h("span", { class: "note", text: state.llm.reason }));
    parts.push(status);

    if (this.discovered.length > 0) {
      for (const server of this.discovered) parts.push(this.serverCard(server));
    } else if (this.scanned) {
      // Only after an actual scan. Saying "nothing found" before looking is a
      // claim the panel has not earned.
      parts.push(h("p", { class: "note", text: s.localNoneFound }));
      parts.push(h("p", { class: "note", text: s.localInstallHint }));
      const links = h("div", { class: "row", style: "gap:10px;margin:6px 0 12px;flex-wrap:wrap" });
      for (const server of state.known_servers) {
        links.append(
          h("a", {
            class: "note",
            href: server.docs_url,
            target: "_blank",
            rel: "noreferrer noopener",
            text: server.label + " :" + server.port,
          }),
        );
      }
      parts.push(links);
    }

    const scan = h("button", {
      class: "btn ghost",
      type: "button",
      text: this.scanning ? s.localScanning : s.localScan,
    }) as HTMLButtonElement;
    scan.disabled = this.scanning;
    scan.addEventListener("click", () => void this.scanLocal());
    parts.push(h("div", { class: "row", style: "margin-bottom:14px" }, scan));

    parts.push(
      this.toggle(s.localPreferLabel, state.llm.prefer, (on) =>
        void this.saveLocal({ prefer: on }),
      ),
    );
    parts.push(h("p", { class: "note", style: "margin:-4px 0 14px", text: s.localPreferHint }));

    parts.push(h("p", { class: "label", text: s.localBaseUrl }));
    const baseUrl = h("input", {
      class: "input",
      type: "text",
      spellcheck: "false",
      autocomplete: "off",
      placeholder: "localhost:11434",
      "aria-label": s.localBaseUrl,
    }) as HTMLInputElement;
    baseUrl.value = state.llm.base_url;

    const model = h("input", {
      class: "input",
      type: "text",
      spellcheck: "false",
      autocomplete: "off",
      placeholder: "qwen2.5:7b",
      "aria-label": s.localModelLabel,
    }) as HTMLInputElement;
    model.value = state.llm.model;

    const apply = h("button", { class: "btn", type: "button", text: s.localApply });
    apply.addEventListener("click", () => {
      void this.saveLocal({ base_url: baseUrl.value, model: model.value });
    });

    parts.push(baseUrl);
    parts.push(h("p", { class: "note", style: "margin:4px 0 10px", text: s.localBaseUrlHint }));
    parts.push(h("p", { class: "label", text: s.localModelLabel }));
    parts.push(model);
    parts.push(h("div", { class: "row", style: "margin:10px 0 20px" }, apply));

    // ---- speech recognition ---------------------------------------------
    parts.push(
      h("p", {
        class: "label",
        style: "padding-top:16px;border-top:1px solid var(--border)",
        text: s.localAsr,
      }),
    );

    if (!state.asr.installed) {
      // The library is absent, so the toggle would set something that cannot
      // take effect. Say why rather than offering a dead control.
      parts.push(h("p", { class: "note", text: s.localAsrMissing }));
      return parts;
    }

    parts.push(
      this.toggle(s.localAsrEnable, state.asr.enabled, (on) =>
        void this.saveLocal({ asr_enabled: on }),
      ),
    );
    parts.push(h("p", { class: "note", style: "margin:-4px 0 14px", text: s.localAsrHint }));

    if (state.asr.enabled) {
      parts.push(h("p", { class: "label", text: s.localSize }));
      for (const size of state.asr.sizes) parts.push(this.sizeCard(size, state.asr.model));
      parts.push(h("p", { class: "note", style: "margin-top:8px", text: s.localSizeHint }));
    }

    return parts;
  }

  private serverCard(server: DiscoveredServer): HTMLElement {
    const s = this.strings;
    const card = h("div", { class: "card" });
    const head = h("div", { class: "row", style: "gap:8px;align-items:center" });
    head.append(h("strong", { text: server.label }));
    if (server.local) {
      // Claimed only for loopback. A model server on the LAN is a perfectly
      // good deployment and a different promise, and the panel must not make
      // the stronger one on its behalf.
      head.append(h("span", { class: "chip", text: s.localOnDevice }));
    }
    card.append(head);

    if (server.models.length === 0) {
      card.append(h("p", { class: "note", text: server.note }));
      return card;
    }

    for (const entry of server.models) {
      const row = h("div", { class: "row", style: "gap:8px;margin-top:6px" });
      const pick = h("button", { class: "btn ghost", type: "button", text: entry.id });
      pick.addEventListener("click", () => {
        void this.saveLocal({ base_url: server.base_url, model: entry.id });
      });
      row.append(pick);
      // Parameter count and quantization are what say whether a model fits in
      // this machine's RAM, which is the question being asked here.
      const detail = [entry.parameters, entry.quantization].filter(Boolean).join(" \u00b7 ");
      if (detail) row.append(h("span", { class: "note", text: detail }));
      card.append(row);
    }
    return card;
  }

  private sizeCard(size: WhisperSize, selected: string): HTMLElement {
    const s = this.strings;
    const on = size.id === selected;
    const card = h("button", { class: "card picker" + (on ? " on" : ""), type: "button" });
    const head = h("div", { class: "row", style: "gap:8px;align-items:center" });
    head.append(h("strong", { text: size.label }));
    if (!size.suitable_ja) {
      // koe is a bilingual tool. Presenting these as a neutral speed slider
      // would mislead in exactly the case it exists for.
      head.append(h("span", { class: "chip warn", text: s.localNotForJa }));
    }
    card.append(head);
    const seconds = Math.round(size.typical_rtf * 60);
    card.append(
      h("p", {
        class: "note",
        text: size.download_mb + " MB " + s.localDownload + " \u00b7 ~" + seconds + "s " + s.localSpeed,
      }),
    );
    card.addEventListener("click", () => void this.saveLocal({ asr_model: size.id }));
    return card;
  }

  private async loadLocal(): Promise<void> {
    try {
      const response = await fetch("/v1/local");
      if (response.ok) this.local = (await response.json()) as LocalState;
    } catch {
      /* the section says it could not load rather than rendering empty */
    }
  }

  private async scanLocal(): Promise<void> {
    this.scanning = true;
    this.render();
    try {
      const response = await fetch("/v1/local/discover", { method: "POST" });
      if (response.ok) {
        const sweep = (await response.json()) as { servers: DiscoveredServer[] };
        this.discovered = sweep.servers;
      }
    } catch {
      this.discovered = [];
    } finally {
      this.scanning = false;
      this.scanned = true;
      await this.loadLocal();
      this.render();
    }
  }

  private async saveLocal(patch: Record<string, unknown>): Promise<void> {
    try {
      const response = await fetch("/v1/local", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (response.ok) this.local = (await response.json()) as LocalState;
    } catch {
      /* the reason line keeps whatever the server last resolved */
    }
    this.render();
  }

  private vocabularySection(): HTMLElement[] {
    const s = this.strings;
    const state = this.vocabulary;
    const parts: HTMLElement[] = [];

    if (!state) {
      parts.push(h("p", { class: "note", text: s.cannotConnect }));
      return parts;
    }
    if (!state.enabled) {
      parts.push(h("p", { class: "note", text: s.vocabularyOff }));
      return parts;
    }

    parts.push(h("p", { class: "note", style: "margin:0 0 6px", text: s.vocabularyHint }));
    parts.push(h("p", { class: "note", style: "margin:0 0 14px", text: s.vocabularyFormat }));

    const editor = h("textarea", {
      class: "vocab-editor",
      rows: "12",
      spellcheck: "false",
      autocapitalize: "off",
      autocomplete: "off",
      placeholder: s.vocabularyPlaceholder,
      "aria-label": s.secVocabulary,
    }) as HTMLTextAreaElement;
    editor.value = this.vocabularyDraft ?? state.text;

    const status = h("span", {
      class: "note",
      text: `${state.entries} ${s.vocabularyCount}`,
    });
    const save = h("button", { class: "btn", type: "button", text: s.vocabularySave });

    // Tracked without re-rendering: rebuilding the panel on every keystroke
    // would move the caret to the end of the box on every keystroke.
    editor.addEventListener("input", () => {
      this.vocabularyDraft = editor.value;
      save.disabled = false;
    });
    save.addEventListener("click", () => void this.saveVocabulary(editor, save, status));

    const actions = h("div", { class: "row", style: "margin-top:12px;align-items:center;gap:10px" });
    actions.append(save, status);

    parts.push(editor, actions);
    return parts;
  }

  private async saveVocabulary(
    editor: HTMLTextAreaElement,
    save: HTMLButtonElement,
    status: HTMLElement,
  ): Promise<void> {
    const s = this.strings;
    save.disabled = true;
    try {
      const response = await fetch("/v1/vocabulary", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: editor.value }),
      });
      if (!response.ok) {
        status.textContent = s.vocabularyOff;
        return;
      }
      const saved = (await response.json()) as VocabularyState;
      this.vocabulary = saved;
      // Adopt the canonical text the server stored, so the box shows what was
      // actually saved rather than what was typed at it.
      this.vocabularyDraft = saved.text;
      editor.value = saved.text;
      status.textContent = `${s.vocabularySaved} · ${saved.entries} ${s.vocabularyCount}`;
    } catch {
      status.textContent = s.cannotConnect;
      save.disabled = false;
    }
  }

  private pluginsSection(): HTMLElement[] {
    const s = this.strings;
    const listing = this.pluginListing;
    const parts: HTMLElement[] = [];

    if (!listing) {
      parts.push(h("p", { class: "note", text: s.cannotConnect }));
      return parts;
    }

    parts.push(h("p", { class: "note", style: "margin:0 0 14px", text: s.pluginsHint }));

    for (const plugin of listing.plugins) {
      // Which tools each plugin contributes, because "what will I lose by
      // turning this off" is the only question anyone has on this screen.
      const provided = this.toolRecords
        .filter((tool) => tool.source === pluginSource(plugin.name))
        .map((tool) => tool.name);
      parts.push(this.pluginCard(plugin, provided));
    }

    const actions = h("div", { class: "row", style: "margin-top:14px" });
    const reload = h("button", { class: "btn ghost", type: "button", text: s.pluginReload });
    reload.addEventListener("click", () => void this.reloadPlugins(reload));
    actions.append(reload);
    parts.push(actions);

    if (listing.directory) {
      parts.push(
        h("p", { class: "label", style: "margin-top:16px", text: s.pluginsDirectory }),
        h("p", { class: "note", style: "font-family:var(--font-mono);font-size:11px", text: listing.directory }),
      );
    }
    parts.push(h("p", { class: "note callout", text: s.pluginsTrust }));
    return parts;
  }

  private pluginCard(plugin: PluginRecord, provided: string[]): HTMLElement {
    const s = this.strings;
    const card = h("div", { class: "plugin" });

    const head = h("div", { class: "plugin-head" });
    const toggle = h("input", { type: "checkbox", class: "switch" }) as HTMLInputElement;
    toggle.checked = plugin.enabled;
    toggle.disabled = Boolean(plugin.error);
    toggle.setAttribute("aria-label", plugin.name);
    toggle.addEventListener("change", () => void this.setPlugin(plugin.name, toggle));

    head.append(toggle, h("span", { class: "plugin-name", text: plugin.name }));
    if (plugin.version) head.append(h("span", { class: "note", text: plugin.version }));
    if (plugin.builtin) head.append(h("span", { class: "chip", text: s.pluginBuiltin }));
    if (plugin.error) head.append(h("span", { class: "chip bad", text: s.pluginFailed }));
    else if (plugin.enabled && !plugin.active) {
      // Enabled but not running means it is waiting on a service it declared
      // — worth distinguishing from broken.
      head.append(h("span", { class: "chip warn", text: s.pluginInactive }));
    }
    card.append(head);

    if (plugin.description) {
      card.append(h("p", { class: "note", style: "margin:8px 0 0", text: plugin.description }));
    }
    if (provided.length > 0) {
      card.append(
        h("p", {
          class: "note",
          style: "margin:6px 0 0;font-family:var(--font-mono);font-size:11px",
          text: `${s.toolsProvided}: ${provided.join(", ")}`,
        }),
      );
    }
    // The traceback, verbatim: a plugin that failed to import is useless to
    // debug without it, and this is the only place it surfaces.
    if (plugin.error) card.append(h("pre", { class: "plugin-error", text: plugin.error }));
    return card;
  }

  private async setPlugin(name: string, toggle: HTMLInputElement): Promise<void> {
    toggle.disabled = true;
    try {
      const response = await fetch(`/v1/plugins/${encodeURIComponent(name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: toggle.checked }),
      });
      if (!response.ok) throw new Error(String(response.status));
      const body = (await response.json()) as { tools: ToolRecord[] };
      this.toolRecords = body.tools;
      await this.loadPlugins();
      this.render();
    } catch {
      this.notify.error(this.strings.cannotConnect);
      toggle.checked = !toggle.checked;
    } finally {
      toggle.disabled = false;
    }
  }

  private async reloadPlugins(button: HTMLButtonElement): Promise<void> {
    button.disabled = true;
    try {
      const response = await fetch("/v1/plugins/reload", { method: "POST" });
      if (!response.ok) throw new Error(String(response.status));
      this.pluginListing = (await response.json()) as PluginListing;
      await this.loadPlugins();
      this.render();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      button.disabled = false;
    }
  }

  /* ------------------------------------------------------------- widgets */

  private heading(text: string): HTMLElement {
    return h("h3", { class: "pane-title", style: "margin:18px 0 8px", text });
  }

  private toggle(label: string, on: boolean, onChange: (on: boolean) => void): HTMLElement {
    const row = h("label", { class: "switch-row" });
    const input = h("input", { type: "checkbox", class: "switch" }) as HTMLInputElement;
    input.checked = on;
    input.addEventListener("change", () => onChange(input.checked));
    row.append(input, h("span", { text: label }));
    return row;
  }

  private slider(
    value: number,
    limits: { min: number; max: number; step: number },
    format: (value: number) => string,
    onChange: (value: number) => void,
  ): HTMLElement {
    const row = h("div", { class: "slider-row" });
    const input = h("input", {
      type: "range",
      class: "range",
      min: String(limits.min),
      max: String(limits.max),
      step: String(limits.step),
      value: String(value),
    }) as HTMLInputElement;
    const readout = h("span", { class: "readout", text: format(value) });
    input.addEventListener("input", () => {
      const next = Number(input.value);
      readout.textContent = format(next);
      onChange(next);
    });
    row.append(input, readout);
    return row;
  }

  /* --------------------------------------------------------- credentials */

  private providerCard(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const card = h("div", { class: "provider" });

    const head = h("div", { class: "provider-head" });
    head.append(h("span", { class: "provider-name", text: provider.label }));

    if (provider.configured) {
      head.append(h("span", { class: "provider-key", text: provider.fingerprint }));
      head.append(this.statusChip(provider));
    } else {
      head.append(h("span", { class: "chip", text: s.notConfigured }));
    }
    if (provider.source === "environment") {
      head.append(h("span", { class: "chip", text: s.fromEnvironment }));
    }
    if (!provider.available) {
      head.append(h("span", { class: "chip warn", text: s.sdkMissing }));
    }
    card.append(head);

    // Said before the key field, not after a failed request: a key that is
    // accepted and then quietly ignored is the confusing case.
    if (!provider.available) {
      card.append(h("p", { class: "note", style: "margin:0 0 8px", text: s.sdkMissingHint }));
    }

    const outcome = this.outcome(provider);
    if (outcome && provider.code !== "sdk_missing") {
      card.append(h("p", { class: "note", style: "margin:0 0 8px", text: outcome }));
    }

    if (!provider.editable) {
      card.append(h("p", { class: "note", text: `${s.fromEnvironmentHint} (${provider.env_var})` }));
      card.append(this.actions(provider));
      return card;
    }

    const input = h("input", {
      class: "input",
      type: "password",
      placeholder: s.keyPlaceholder,
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": `${provider.label} ${s.apiKeys}`,
    }) as HTMLInputElement;

    const save = h("button", { class: "btn", type: "button", text: s.save }) as HTMLButtonElement;
    save.addEventListener("click", () => void this.save(provider, input, save));
    input.addEventListener("keydown", (event) => {
      if ((event as KeyboardEvent).key === "Enter") void this.save(provider, input, save);
    });

    card.append(h("div", { class: "row", style: "margin-bottom:8px" }, input, save));
    card.append(this.actions(provider));
    return card;
  }

  private statusChip(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const map = {
      valid: { cls: "chip live", text: s.statusValid },
      invalid: { cls: "chip bad", text: s.statusInvalid },
      error: { cls: "chip warn", text: s.statusError },
      unknown: { cls: "chip", text: s.statusUnknown },
    } as const;
    const entry = map[provider.status] ?? map.unknown;
    return h("span", { class: entry.cls, text: entry.text });
  }

  private actions(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const row = h("div", { class: "row" });

    const test = h("button", {
      class: "btn ghost",
      type: "button",
      text: s.test,
    }) as HTMLButtonElement;
    test.disabled = !provider.configured;
    test.addEventListener("click", () => void this.verify(provider, test));
    row.append(test);

    if (provider.editable && provider.configured) {
      const remove = h("button", { class: "btn ghost", type: "button", text: s.remove });
      remove.addEventListener("click", () => void this.remove(provider));
      row.append(remove);
    }

    row.append(h("div", { style: "flex:1" }));
    row.append(
      h("a", {
        class: "note",
        href: provider.docs_url,
        target: "_blank",
        rel: "noopener noreferrer",
        text: s.getKey,
      }),
    );
    return row;
  }

  /**
   * A verification outcome in the reader's language.
   *
   * Falls back to the server's English detail when the outcome carried a
   * vendor exception rather than one of the enumerated codes — a specific
   * English message beats a vague translated one.
   */
  private outcome(provider: ProviderCredential): string {
    const s = this.strings;
    const known: Record<string, string> = {
      no_key: s.vNoKey,
      accepted: s.vAccepted,
      rejected: s.vRejected,
      no_permission: s.vNoPermission,
      rate_limited: s.vRateLimited,
      unreachable: s.vUnreachable,
      sdk_missing: s.sdkMissingHint,
    };
    return known[provider.code] ?? provider.detail;
  }

  /** Render a server rejection in the reader's language, or fall back to it. */
  private explain(error: ServerError, status: number): string {
    const s = this.strings;
    const context = error.context ?? {};
    const template =
      error.code === "bad_prefix"
        ? s.errBadPrefix
        : error.code === "empty_key"
          ? s.errEmptyKey
          : error.code === "env_managed"
            ? s.errEnvManaged
            : "";
    if (!template) return error.detail ?? `HTTP ${status}`;
    return template.replace(/\{(\w+)\}/g, (whole, key: string) => context[key] ?? whole);
  }

  private async save(
    provider: ProviderCredential,
    input: HTMLInputElement,
    button: HTMLButtonElement,
  ): Promise<void> {
    const key = input.value.trim();
    if (!key) return;

    button.disabled = true;
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const body = (await response.json().catch(() => ({}))) as ServerError;
      if (!response.ok) {
        input.setAttribute("aria-invalid", "true");
        this.notify.error(this.explain(body, response.status));
        return;
      }
      // Clear immediately: there is no reason for a secret to sit in a DOM
      // node after it has been accepted.
      input.value = "";
      input.removeAttribute("aria-invalid");
      this.notify.ok(`${provider.label} — ${this.strings.saved}`);
      this.host.onCredentialsChanged();
      await this.loadCredentials();
      this.render();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      button.disabled = false;
    }
  }

  private async verify(provider: ProviderCredential, button: HTMLButtonElement): Promise<void> {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = this.strings.testing;
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}/verify`, {
        method: "POST",
      });
      const body = await response.json();
      if (!response.ok || !body.provider) {
        this.notify.error(this.explain(body as ServerError, response.status));
        return;
      }
      const updated = body.provider as ProviderCredential;
      const message = `${provider.label} — ${this.outcome(updated)}`;
      if (updated.status === "valid") this.notify.ok(message);
      else if (updated.status === "invalid") this.notify.error(message);
      else this.notify.show(message, "info");
      await this.loadCredentials();
      this.render();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  private async remove(provider: ProviderCredential): Promise<void> {
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}`, { method: "DELETE" });
      if (!response.ok) throw new Error(String(response.status));
      this.notify.ok(`${provider.label} — ${this.strings.removed}`);
      this.host.onCredentialsChanged();
      await this.loadCredentials();
      this.render();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    }
  }
}

/** The keyboard-shortcut help sheet. */
export class ShortcutsDialog {
  private readonly dialog: HTMLDialogElement;
  private readonly body: HTMLElement;

  constructor(private strings: Strings) {
    this.dialog = h("dialog", { class: "modal", "aria-labelledby": "shortcuts-title" });
    this.body = h("div", { class: "modal-body" });
    const head = h(
      "div",
      { class: "modal-head" },
      h("h2", { class: "modal-title", id: "shortcuts-title", text: strings.shortcuts }),
    );
    const foot = h("div", { class: "modal-foot" });
    const close = h("button", { class: "btn", type: "button", text: strings.close });
    close.addEventListener("click", () => this.dialog.close());
    foot.append(h("div", { style: "flex:1" }), close);
    this.dialog.append(head, this.body, foot);
    document.body.append(this.dialog);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
  }

  open(): void {
    const s = this.strings;
    const grid = h("div", { class: "keys" });
    const rows: Array<[string, string]> = [
      ["Ctrl+K", s.paletteSearch],
      ["Ctrl+1…4", s.shortcutWorkspace],
      ["Space", s.toggleRecord],
      ["/", s.shortcutSearch],
      [",", s.shortcutSettings],
      ["?", s.shortcutHelp],
      ["Esc", s.shortcutClose],
    ];
    for (const [key, description] of rows) {
      grid.append(h("kbd", { text: key }), h("span", { text: description }));
    }
    this.body.replaceChildren(grid);
    this.dialog.showModal();
  }
}


/**
 * The `source` a plugin's tools declare, from its name.
 *
 * They differ because a plugin is named for what it is ("workspace-tools")
 * and a tool's source for where it came from ("workspace"). Deriving one from
 * the other here beats making every tool restate its plugin's full name.
 */
function pluginSource(name: string): string {
  return name.replace(/-tools$/, "");
}
