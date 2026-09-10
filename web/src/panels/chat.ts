/**
 * One session's conversation.
 *
 * Laid out after Claude Code's desktop app, because what an agent produces is
 * not a chat. A messaging layout is built for two people taking turns; an
 * agent turn is mostly *actions* with a little prose between them, and the
 * reader's question is "what did it do, and did it work". So:
 *
 *   - **The conversation is the page.** A centred reading column: your
 *     messages in a bubble on the right, the agent's answers as plain prose.
 *     No avatars — there are only ever two parties, and both are obvious.
 *   - **Tool calls are rows, not prose.** "Read README.md", "Ran npm test":
 *     a status icon, a verb, a target, a duration, and the output one click
 *     away. The calls of one step share a card, so a burst of reads reads as
 *     one burst.
 *   - **While it works, a working line.** Elapsed time and step number,
 *     because a turn doing six model calls and a turn waiting on one slow one
 *     look identical behind a spinner, and they call for different patience.
 *   - **The composer steers.** Typing while a turn runs joins it at the next
 *     step boundary instead of queueing behind it, and an empty composer's
 *     send button becomes stop. Esc interrupts.
 *   - **Context is a ring beside the model.** The number that explains both
 *     "why did it forget that" and "why is this costing so much", where you
 *     are already looking when you notice either.
 */

import { copyText, h } from "../dom";
import type { Strings } from "../i18n";
import { icon, type IconName } from "../icons";
import { renderMarkdown } from "../markdown";
import { IDLE_AGENT, type AgentActivity, type SlashCommand } from "../state";
import type { Notifications } from "../toast";

export interface ContextGauge {
  total: number;
  window: number;
  pressure: number;
}

export interface ChatEvents {
  onActivity?: (activity: AgentActivity) => void;
  onTitle?: (title: string) => void;
  onModelClick?: () => void;
}

interface ToolView {
  row: HTMLElement;
  head: HTMLElement;
  status: HTMLElement;
  time: HTMLElement;
  body: HTMLElement;
  output: HTMLElement;
  verb: string;
}

const HISTORY_LIMIT = 50;
const SVG = "http://www.w3.org/2000/svg";
/** Circumference of the context ring (r = 7). */
const RING = 2 * Math.PI * 7;

export class ChatPanel {
  private socket: WebSocket | null = null;
  private readonly root: HTMLElement;
  private readonly scroller: HTMLElement;
  private readonly column: HTMLElement;
  private readonly working: HTMLElement;
  private readonly workingText: HTMLElement;
  private readonly workingMeta: HTMLElement;
  private readonly greeting: HTMLElement;
  private readonly suggestions: HTMLElement;
  private readonly composer: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly menu: HTMLElement;
  private readonly pop: HTMLElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly ring: HTMLButtonElement;
  private readonly ringArc: SVGCircleElement;
  private readonly modelChip: HTMLButtonElement;
  private readonly steerHint: HTMLElement;
  private readonly foot: HTMLElement;

  private activity: AgentActivity = IDLE_AGENT;
  private gaugeState: ContextGauge = { total: 0, window: 0, pressure: 0 };
  private adapter = "";
  private fallbackModel = "";
  private title = "";
  private turnStarted = 0;
  private ticker = 0;
  private toolGroup: HTMLElement | null = null;
  private runningTool = "";
  private sendMode: "send" | "stop" | "" = "";
  private readonly calls = new Map<string, ToolView>();

  private commands: readonly SlashCommand[] = [];
  private menuItems: SlashCommand[] = [];
  private menuActive = 0;

  private readonly history: string[] = [];
  private historyAt = -1;
  private hadConversation = false;
  private disposed = false;
  /** Whether the reader was at the bottom, measured before anything is added. */
  private pinned = true;

  /**
   * Closes the popovers when a press lands anywhere else on the page.
   *
   * On the document rather than this panel: a listener on the panel never
   * heard a click in the side panel, so the context card stayed open over the
   * conversation while someone started recording beside it.
   */
  private readonly onOutside = (event: PointerEvent): void => {
    const target = event.target as Node;
    if (!this.pop.hidden && !this.pop.contains(target) && !this.ring.contains(target)) {
      this.pop.hidden = true;
    }
    if (!this.menu.hidden && !this.composer.contains(target)) this.menu.hidden = true;
  };

  constructor(
    host: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
    private readonly events: ChatEvents = {},
  ) {
    this.column = h("div", { class: "chat-column", role: "log", "aria-live": "polite" });
    this.scroller = h("div", { class: "chat-scroll" }, this.column);

    this.workingText = h("span", { class: "shimmer" });
    this.workingMeta = h("span", { class: "working-meta" });
    this.working = h(
      "div",
      { class: "working", role: "status" },
      h("span", { class: "spark" }, icon("spark", 16)),
      this.workingText,
      this.workingMeta,
    );
    this.working.hidden = true;
    this.column.append(this.working);

    this.greeting = h("div", { class: "greeting" });
    this.suggestions = h("div", { class: "suggestions" });

    this.input = h("textarea", { class: "composer-input", rows: "1", spellcheck: "false" });
    this.menu = h("div", { class: "cmenu", role: "listbox" });
    this.menu.hidden = true;
    this.pop = h("div", { class: "ctx-pop", role: "dialog" });
    this.pop.hidden = true;

    const slash = h("button", { class: "cbtn", type: "button" }, icon("slash", 16));
    slash.addEventListener("click", () => this.insert("/"));
    slash.dataset.role = "commands";

    this.steerHint = h("span", { class: "steer-hint" });
    this.steerHint.hidden = true;

    const ring = ringSvg();
    this.ringArc = ring.arc;
    this.ring = h("button", { class: "ctx-ring", type: "button" });
    this.ring.append(ring.svg);
    this.ring.hidden = true;

    this.modelChip = h("button", { class: "model-chip", type: "button" });
    this.sendButton = h("button", { class: "send", type: "button" });

    this.composer = h(
      "div",
      { class: "composer" },
      this.menu,
      this.pop,
      this.input,
      h(
        "div",
        { class: "composer-bar" },
        slash,
        this.steerHint,
        h("span", { class: "grow" }),
        this.ring,
        this.modelChip,
        this.sendButton,
      ),
    );
    this.foot = h("p", { class: "composer-foot" });

    this.root = h(
      "div",
      { class: "chat is-empty" },
      this.scroller,
      h(
        "div",
        { class: "chat-dock" },
        h("div", { class: "dock-column" }, this.greeting, this.composer, this.foot, this.suggestions),
      ),
    );
    host.append(this.root);

    this.bind();
    this.renderGreeting();
    this.renderComposer();
    // Opened at once: the server answers the command list before any
    // conversation exists, so completion works for the first thing typed.
    this.connect();
  }

  /* ---------------------------------------------------------------- public */

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.renderGreeting();
    this.renderComposer();
    if (!this.pop.hidden) this.renderPop();
    if (this.activity.busy) this.tick();
  }

  /** The app's idea of the model, until the server names the one it used. */
  setModel(name: string): void {
    if (name === this.fallbackModel) return;
    this.fallbackModel = name;
    this.renderComposer();
  }

  focus(): void {
    this.input.focus();
  }

  /**
   * Quote a selection into the composer, as ⌘L does in an editor.
   *
   * Quoted rather than silently attached, so what the model will be shown is
   * exactly what is in the box and can be trimmed before sending.
   */
  quote(text: string): void {
    const quoted = text
      .trim()
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");
    const current = this.input.value.trim();
    this.input.value = current ? `${quoted}\n\n${current}` : `${quoted}\n\n`;
    this.afterEdit();
    this.input.focus();
    this.input.setSelectionRange(this.input.value.length, this.input.value.length);
  }

  /** Put text in the composer without sending it. */
  insert(text: string): void {
    this.input.value = text;
    this.afterEdit();
    this.input.focus();
    this.input.setSelectionRange(text.length, text.length);
  }

  dispose(): void {
    this.disposed = true;
    document.removeEventListener("pointerdown", this.onOutside, true);
    window.clearInterval(this.ticker);
    this.socket?.close();
    this.socket = null;
    this.root.remove();
  }

  /* ---------------------------------------------------------------- input */

  private bind(): void {
    this.scroller.addEventListener(
      "scroll",
      () => {
        const distance = this.scroller.scrollHeight - this.scroller.scrollTop - this.scroller.clientHeight;
        this.pinned = distance < 80;
      },
      { passive: true },
    );
    this.input.addEventListener("keydown", (event) => this.onKey(event));
    this.input.addEventListener("input", () => {
      this.historyAt = -1;
      this.afterEdit();
    });
    this.composer.addEventListener("click", (event) => {
      if (event.target === this.composer) this.input.focus();
    });
    this.sendButton.addEventListener("click", () => {
      if (this.sendMode === "stop") this.cancel();
      else this.submit();
    });
    this.modelChip.addEventListener("click", () => this.events.onModelClick?.());
    this.ring.addEventListener("click", () => {
      this.pop.hidden = !this.pop.hidden;
      if (!this.pop.hidden) this.renderPop();
    });
    this.menu.addEventListener("mousedown", (event) => {
      // mousedown, not click: a click would blur the input first and close
      // the menu out from under the pointer.
      const item = (event.target as HTMLElement).closest<HTMLElement>("[data-command]");
      if (!item) return;
      event.preventDefault();
      this.runCommand(item.dataset.command ?? "");
    });
    document.addEventListener("pointerdown", this.onOutside, true);
  }

  private afterEdit(): void {
    this.grow();
    this.updateMenu();
    this.updateSend();
  }

  private onKey(event: KeyboardEvent): void {
    const menuOpen = !this.menu.hidden && this.menuItems.length > 0;

    if (menuOpen && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      this.menuActive = (this.menuActive + step + this.menuItems.length) % this.menuItems.length;
      this.renderMenu();
      return;
    }
    if (menuOpen && event.key === "Tab") {
      event.preventDefault();
      this.insert(`/${this.menuItems[this.menuActive].name} `);
      return;
    }
    if (menuOpen && event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      this.runCommand(this.menuItems[this.menuActive].name);
      return;
    }

    if (event.key === "Escape") {
      if (menuOpen) this.menu.hidden = true;
      else if (!this.pop.hidden) this.pop.hidden = true;
      else if (this.activity.busy) this.cancel();
      else return;
      // Handled here, so the page's own Escape does not also fire behind it.
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      // isComposing: Enter confirms a kana-to-kanji conversion, and sending the
      // half-converted sentence is the most irritating thing a composer can do
      // to someone typing Japanese.
      event.preventDefault();
      this.submit();
      return;
    }

    // History, but only from the edges, so arrows still move the caret
    // through a multi-line draft.
    if (event.key === "ArrowUp" && this.caretOnFirstLine() && this.history.length > 0) {
      event.preventDefault();
      this.historyAt = Math.min(this.history.length - 1, this.historyAt + 1);
      this.input.value = this.history[this.history.length - 1 - this.historyAt];
      this.grow();
      this.updateSend();
      return;
    }
    if (event.key === "ArrowDown" && this.historyAt >= 0 && this.caretOnLastLine()) {
      event.preventDefault();
      this.historyAt -= 1;
      this.input.value =
        this.historyAt >= 0 ? this.history[this.history.length - 1 - this.historyAt] : "";
      this.grow();
      this.updateSend();
    }
  }

  private caretOnFirstLine(): boolean {
    return !this.input.value.slice(0, this.input.selectionStart).includes("\n");
  }

  private caretOnLastLine(): boolean {
    return !this.input.value.slice(this.input.selectionEnd).includes("\n");
  }

  private grow(): void {
    this.input.style.height = "auto";
    const height = this.input.scrollHeight;
    this.input.style.height = `${Math.min(240, height)}px`;
    // A scrollbar only once the box has stopped growing; before that it is a
    // pair of arrows beside two lines of placeholder.
    this.input.style.overflowY = height > 240 ? "auto" : "hidden";
  }

  private updateMenu(): void {
    const match = /^\/(\S*)$/.exec(this.input.value);
    if (!match || this.commands.length === 0) {
      this.menu.hidden = true;
      this.menuItems = [];
      return;
    }
    const prefix = match[1].toLowerCase();
    this.menuItems = this.commands.filter((command) => command.name.startsWith(prefix));
    this.menuActive = Math.min(this.menuActive, Math.max(0, this.menuItems.length - 1));
    this.menu.hidden = this.menuItems.length === 0;
    this.renderMenu();
  }

  private renderMenu(): void {
    this.menu.replaceChildren(
      ...this.menuItems.map((command, index) =>
        h(
          "div",
          {
            class: `cmenu-item${index === this.menuActive ? " on" : ""}`,
            role: "option",
            "aria-selected": String(index === this.menuActive),
            "data-command": command.name,
          },
          h("span", { class: "cmenu-name", text: `/${command.name}` }),
          h("span", { class: "cmenu-summary", text: command.summary }),
        ),
      ),
    );
  }

  private runCommand(name: string): void {
    if (!name) return;
    this.input.value = `/${name}`;
    this.menu.hidden = true;
    this.submit();
  }

  /* ---------------------------------------------------------------- socket */

  private connect(): WebSocket {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return this.socket;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/v1/chat/stream`);
    socket.addEventListener("open", () => socket.send(JSON.stringify({ commands: true })));
    socket.addEventListener("message", (event) => {
      if (this.socket === socket) this.onMessage(event);
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      if (this.disposed) return;
      if (this.activity.busy) {
        this.fail(this.strings.cannotConnect);
      } else if (this.hadConversation) {
        // The server's agent went with the socket. Say so, rather than let the
        // next message quietly start a conversation that remembers nothing.
        this.add(h("div", { class: "m-note", text: this.strings.chatConnectionLost }));
      }
      this.hadConversation = false;
    });
    this.socket = socket;
    return socket;
  }

  private transmit(payload: Record<string, unknown>): void {
    const socket = this.connect();
    const data = JSON.stringify(payload);
    if (socket.readyState === WebSocket.OPEN) socket.send(data);
    else socket.addEventListener("open", () => socket.send(data), { once: true });
  }

  private submit(): void {
    const text = this.input.value.trim();
    if (!text) return;

    this.input.value = "";
    this.grow();
    this.menu.hidden = true;
    this.pop.hidden = true;
    if (this.history[this.history.length - 1] !== text) this.history.push(text);
    if (this.history.length > HISTORY_LIMIT) this.history.shift();
    this.historyAt = -1;

    // While a turn is running the composer steers it rather than queueing a
    // new one: a model three tool calls into the wrong file needs telling now,
    // and cancelling to re-prompt throws away the work already done.
    const steering = this.activity.busy;
    const isCommand = /^\/[a-zA-Z][\w-]*(\s|$)/.test(text);
    this.showUser(text, steering);

    if (!steering && !isCommand) {
      if (!this.title) {
        this.title = text.replace(/\s+/g, " ").slice(0, 80);
        this.events.onTitle?.(this.title);
      }
      this.beginTurn();
    }
    this.transmit(steering ? { steer: text } : { message: text });
    this.updateSend();
  }

  private cancel(): void {
    if (!this.activity.busy) return;
    this.transmit({ cancel: true });
    this.workingText.textContent = this.strings.chatStopping;
  }

  /* ---------------------------------------------------------------- frames */

  private onMessage(event: MessageEvent): void {
    const frame = JSON.parse(String(event.data)) as Record<string, unknown>;
    switch (String(frame.type ?? "")) {
      case "conversation":
        this.hadConversation = true;
        this.adapter = String(frame.adapter ?? "");
        this.setActivity({ ...this.activity, model: this.adapter });
        this.renderComposer();
        break;

      case "commands":
        this.commands = (frame.commands as SlashCommand[]) ?? [];
        this.updateMenu();
        break;

      case "step/start":
        this.toolGroup = null;
        this.setActivity({ ...this.activity, step: Number(frame.step ?? 0) });
        this.tick();
        break;

      case "assistant/text":
        this.showAnswer(String(frame.text ?? ""));
        break;

      case "tool/call":
        this.showCall(frame);
        break;

      case "tool/result":
        this.showResult(frame);
        break;

      case "steered": {
        const steers = this.column.querySelectorAll(".m-user.steer");
        steers[steers.length - 1]?.classList.add("landed");
        break;
      }

      case "compaction":
        this.showCompaction(frame);
        break;

      case "command":
        this.showCommand(frame);
        break;

      case "cancelled":
        this.add(h("div", { class: "m-note bad", text: this.strings.chatInterrupted }));
        break;

      case "turn/end":
        this.endTurn(frame);
        break;

      case "turn/error":
      case "error":
        this.fail(String(frame.message ?? "error"));
        break;
    }
  }

  private beginTurn(): void {
    this.calls.clear();
    this.toolGroup = null;
    this.runningTool = "";
    this.turnStarted = performance.now();
    this.setActivity({ ...this.activity, busy: true, step: 0 });
    this.working.hidden = false;
    this.column.append(this.working);
    window.clearInterval(this.ticker);
    this.ticker = window.setInterval(() => this.tick(), 250);
    this.tick();
    this.scroll(true);
  }

  private endTurn(frame: Record<string, unknown>): void {
    const s = this.strings;
    const parts = [s.chatWorkedFor.replace("{t}", elapsed(Number(frame.duration_ms ?? 0)))];
    const calls = Number(frame.tool_calls ?? 0);
    if (calls > 0) parts.push(`${calls} ${calls === 1 ? s.chatToolCall : s.chatToolCalls}`);
    const tokens = Number(frame.input_tokens ?? 0) + Number(frame.output_tokens ?? 0);
    if (tokens > 0) parts.push(`${compactNumber(tokens)} tokens`);
    // Only when it is not the ordinary one: "completed" on every turn is
    // noise, "max-steps" on one of them is the answer to why it stopped.
    const reason = String(frame.reason ?? "");
    if (reason && reason !== "completed" && reason !== "cancelled") parts.push(reason);

    this.add(h("div", { class: "m-footer" }, icon("spark", 12), h("span", { text: parts.join(" · ") })));
    this.gauge(frame.context);
    this.stopWorking();
    this.setActivity({ ...this.activity, turns: this.activity.turns + 1 });
    this.scroll();
  }

  private fail(message: string): void {
    this.add(h("div", { class: "m-error ja", text: message }));
    this.stopWorking();
    this.scroll();
  }

  private stopWorking(): void {
    window.clearInterval(this.ticker);
    this.working.hidden = true;
    this.runningTool = "";
    this.setActivity({ ...this.activity, busy: false, step: 0 });
  }

  private tick(): void {
    const s = this.strings;
    this.workingText.textContent = this.runningTool ? `${this.runningTool}…` : s.chatThinking;
    const parts = [elapsed(performance.now() - this.turnStarted, true)];
    if (this.activity.step > 1) parts.push(`${s.chatStep} ${this.activity.step}`);
    parts.push(s.chatEscInterrupt);
    this.workingMeta.textContent = parts.join(" · ");
  }

  private setActivity(activity: AgentActivity): void {
    const changedBusy = this.activity.busy !== activity.busy;
    this.activity = activity;
    if (changedBusy) this.renderComposer();
    if (!this.pop.hidden) this.renderPop();
    this.events.onActivity?.(activity);
  }

  private gauge(context: unknown): void {
    if (!context || typeof context !== "object") return;
    const data = context as Record<string, unknown>;
    this.gaugeState = {
      total: Number(data.total ?? 0),
      window: Number(data.context_window ?? 0),
      pressure: Number(data.pressure ?? 0),
    };
    this.renderRing();
    if (!this.pop.hidden) this.renderPop();
  }

  /* ---------------------------------------------------------------- render */

  private add(node: HTMLElement): HTMLElement {
    this.column.insertBefore(node, this.working);
    this.setEmpty(false);
    return node;
  }

  private setEmpty(empty: boolean): void {
    if (this.root.classList.contains("is-empty") === empty) return;
    this.root.classList.toggle("is-empty", empty);
    this.renderComposer();
  }

  private renderGreeting(): void {
    const s = this.strings;
    const hour = new Date().getHours();
    const hello =
      hour >= 5 && hour < 11 ? s.greetMorning : hour >= 11 && hour < 18 ? s.greetAfternoon : s.greetEvening;
    this.greeting.replaceChildren(
      h("div", { class: "greeting-mark", "aria-hidden": "true", text: "声" }),
      h("h2", { class: "greeting-title", text: hello }),
      h("p", { class: "greeting-sub ja", text: s.chatEmpty }),
    );

    this.suggestions.replaceChildren(
      ...[
        ["waveform", s.chatSuggest1],
        ["minutes", s.chatSuggest2],
        ["code", s.chatSuggest3],
      ].map(([glyph, text]) => {
        const button = h("button", { class: "suggestion ja", type: "button" }, icon(glyph as IconName, 15), text);
        button.addEventListener("click", () => {
          this.input.value = text;
          this.submit();
        });
        return button;
      }),
    );
  }

  private renderComposer(): void {
    const s = this.strings;
    const busy = this.activity.busy;
    const empty = this.root.classList.contains("is-empty");
    this.input.placeholder = busy
      ? s.chatSteerPlaceholder
      : empty
        ? s.chatPlaceholder
        : s.chatReplyPlaceholder;
    this.input.setAttribute("aria-label", this.input.placeholder);
    this.composer.classList.toggle("steer", busy);
    this.steerHint.hidden = !busy;
    this.steerHint.textContent = s.chatHintBusy;
    this.foot.textContent = s.chatDisclaimer;

    const commands = this.composer.querySelector<HTMLElement>('[data-role="commands"]');
    if (commands) {
      commands.title = `${s.chatCommandsButton} (/)`;
      commands.setAttribute("aria-label", s.chatCommandsButton);
    }

    const model = this.adapter || this.fallbackModel || "—";
    this.modelChip.replaceChildren(h("span", { text: model }), icon("chevron-down", 14));
    this.modelChip.title = s.harnessModel;
    this.updateSend(true);
    this.renderRing();
  }

  private updateSend(force = false): void {
    const s = this.strings;
    const hasText = this.input.value.trim().length > 0;
    const mode = this.activity.busy && !hasText ? "stop" : "send";
    if (force || mode !== this.sendMode) {
      this.sendMode = mode;
      this.sendButton.replaceChildren(icon(mode === "stop" ? "stop" : "arrow-up", 16));
      this.sendButton.classList.toggle("stop", mode === "stop");
    }
    this.sendButton.disabled = mode === "send" && !hasText;
    const label = mode === "stop" ? `${s.chatStop} (Esc)` : this.activity.busy ? s.chatSteer : s.chatSend;
    this.sendButton.title = label;
    this.sendButton.setAttribute("aria-label", label);
  }

  private renderRing(): void {
    const pressure = Math.max(0, Math.min(1, this.gaugeState.pressure));
    this.ring.hidden = this.gaugeState.total <= 0;
    this.ringArc.style.strokeDashoffset = String(RING * (1 - Math.max(pressure, 0.02)));
    this.ring.classList.toggle("warn", pressure >= 0.6 && pressure < 0.8);
    this.ring.classList.toggle("bad", pressure >= 0.8);
    const label = this.strings.contextUsed.replace("{p}", String(Math.round(pressure * 100)));
    this.ring.title = `${label} · ${this.strings.contextAutoCompact}`;
    this.ring.setAttribute("aria-label", label);
  }

  /** The harness in a card: what this session's agent is doing and how full it is. */
  private renderPop(): void {
    const s = this.strings;
    const { total, window: size, pressure } = this.gaugeState;
    const facts = h("dl", { class: "kv" });
    const fact = (label: string, value: string) =>
      facts.append(h("dt", { text: label }), h("dd", { text: value }));
    fact(s.harnessModel, this.adapter || this.fallbackModel || "—");
    fact(
      s.harnessContext,
      size > 0
        ? `${Math.round(pressure * 100)}% · ${compactNumber(total)} / ${compactNumber(size)}`
        : compactNumber(total),
    );
    fact(s.harnessTurns, String(this.activity.turns));
    fact(s.harnessToolCalls, String(this.activity.toolCalls));
    fact(s.harnessCompactions, String(this.activity.compactions));

    const bar = h("i");
    bar.style.width = `${(Math.max(0, Math.min(1, pressure)) * 100).toFixed(1)}%`;
    bar.className = pressure >= 0.8 ? "bad" : pressure >= 0.6 ? "warn" : "";

    const compact = h("button", { class: "btn ghost sm", type: "button" }, icon("compress", 14), s.compactNow);
    compact.disabled = this.activity.busy || !this.hadConversation;
    compact.addEventListener("click", () => {
      this.pop.hidden = true;
      this.input.value = "/compact";
      this.submit();
    });

    this.pop.replaceChildren(
      h("div", { class: "gauge" }, bar),
      facts,
      h("p", { class: "note ja", text: s.harnessHint }),
      compact,
    );
  }

  private showUser(text: string, steering: boolean): void {
    const node = h("div", { class: `m-user${steering ? " steer" : ""}` });
    if (steering) node.append(h("div", { class: "steer-label", text: `↳ ${this.strings.chatSteer}` }));
    node.append(h("div", { class: "bubble ja", text }));
    this.add(node);
    this.toolGroup = null;
    this.scroll(true);
  }

  private showAnswer(text: string): void {
    if (!text.trim()) return;
    this.toolGroup = null;
    const prose = h("div", { class: "prose" });
    prose.append(renderMarkdown(text));

    const copy = h(
      "button",
      { class: "icon-btn ghost sm", type: "button", title: this.strings.copy, "aria-label": this.strings.copy },
      icon("copy", 14),
    );
    copy.addEventListener("click", () => {
      void copyText(text).then((ok) =>
        ok ? this.notify.ok(this.strings.copied) : this.notify.error(this.strings.copyFailed),
      );
    });
    this.add(h("div", { class: "m-assistant" }, prose, h("div", { class: "m-actions" }, copy)));
    this.scroll();
  }

  private showCall(frame: Record<string, unknown>): void {
    const id = String(frame.id ?? "");
    const name = String(frame.name ?? "");
    const args = (frame.arguments ?? {}) as Record<string, unknown>;
    const { verb, target, glyph } = describeTool(name, args, this.strings);

    if (!this.toolGroup) this.toolGroup = this.add(h("div", { class: "tools" }));

    const status = h("span", { class: "tool-status" }, h("span", { class: "spinner" }));
    const time = h("span", { class: "tool-time" });
    const head = h(
      "button",
      { class: "tool-head", type: "button", "aria-expanded": "false", title: `${name}(${describeArgs(args, 400)})` },
      status,
      h("span", { class: "tool-glyph" }, icon(glyph, 14)),
      h("span", { class: "tool-verb", text: verb }),
      h("span", { class: "tool-target", text: target }),
      h("span", { class: "grow" }),
      time,
      h("span", { class: "tool-chev" }, icon("chevron-right", 14)),
    );
    const output = h("pre", { class: "tool-output" });
    const body = h("div", { class: "tool-body" }, output);
    body.hidden = true;
    const row = h("div", { class: "tool running" }, head, body);
    head.addEventListener("click", () => {
      const open = body.hidden;
      body.hidden = !open;
      head.setAttribute("aria-expanded", String(open));
      row.classList.toggle("open", open);
    });

    this.toolGroup.append(row);
    this.calls.set(id, { row, head, status, time, body, output, verb });
    this.runningTool = verb;
    this.setActivity({ ...this.activity, toolCalls: this.activity.toolCalls + 1 });
    this.tick();
    this.scroll();
  }

  private showResult(frame: Record<string, unknown>): void {
    // The harness keys a result by `id`, the call it answers; `call_id` is the
    // pre-harness loop's name for it, accepted so an older server still pairs.
    const view = this.calls.get(String(frame.id ?? frame.call_id ?? ""));
    if (!view) return;

    const ok = Boolean(frame.ok);
    const ms = Number(frame.duration_ms ?? 0);
    view.row.classList.remove("running");
    view.row.classList.add(ok ? "ok" : "failed");
    view.status.replaceChildren(icon(ok ? "check" : "x", 14));
    // The error code, not just "failed": `no_tool` and `denied` send a reader
    // to completely different places.
    view.time.textContent = ok ? elapsed(ms) : `${String(frame.error ?? "error")} · ${elapsed(ms)}`;
    view.output.textContent = String(frame.content ?? "").replace(/\s+$/, "") || this.strings.chatNoOutput;
    if (!ok) {
      // A failure opens itself: it is the thing the reader needs.
      view.body.hidden = false;
      view.head.setAttribute("aria-expanded", "true");
      view.row.classList.add("open");
    }
    if (this.runningTool === view.verb) this.runningTool = "";
    this.tick();
    this.scroll();
  }

  private showCompaction(frame: Record<string, unknown>): void {
    const error = frame.error ? String(frame.error) : "";
    const shadowed = Number(frame.shadowed ?? 0);
    const tokens = Number(frame.shadowed_tokens ?? 0);
    const detail =
      error ||
      [shadowed > 0 ? `${shadowed} events` : "", tokens > 0 ? `${compactNumber(tokens)} tokens` : ""]
        .filter(Boolean)
        .join(" · ");
    this.add(
      h(
        "div",
        { class: `m-divider${error ? " bad" : ""}` },
        h("span", {}, icon("compress", 13), `${this.strings.chatCompacted}${detail ? ` · ${detail}` : ""}`),
      ),
    );
    if (!error) this.setActivity({ ...this.activity, compactions: this.activity.compactions + 1 });
    this.scroll();
  }

  /**
   * A command's answer.
   *
   * Rendered as the harness replying, not as the assistant: a transcript in
   * which the model answered /context is a transcript of something that did
   * not happen.
   */
  private showCommand(frame: Record<string, unknown>): void {
    this.gauge(frame.context);
    if (frame.reload) {
      this.reset();
      return;
    }
    this.add(h("div", { class: `m-command${frame.ok ? "" : " bad"}` }, h("pre", { text: String(frame.text ?? "") })));
    this.scroll();
  }

  /** /clear: the same session, a new conversation. */
  private reset(): void {
    for (const node of [...this.column.children]) if (node !== this.working) node.remove();
    this.calls.clear();
    this.toolGroup = null;
    this.title = "";
    this.events.onTitle?.("");
    this.setActivity({ ...IDLE_AGENT, model: this.adapter });
    this.setEmpty(true);
  }

  private scroll(force = false): void {
    // Follow the output only if the reader was at the bottom *before* this
    // entry arrived. Measuring after appending made any long answer "far from
    // the bottom", so the log stopped following exactly when it mattered; and
    // yanking someone down while they re-read a result makes it unreadable.
    if (force) this.pinned = true;
    if (this.pinned) this.scroller.scrollTop = this.scroller.scrollHeight;
  }
}

/* -------------------------------------------------------------------- tools */

/**
 * A call the way a person would say it: "Read README.md", not
 * `read_file(path="README.md")`. The raw call is still on the row's tooltip.
 */
function describeTool(
  name: string,
  args: Record<string, unknown>,
  s: Strings,
): { verb: string; target: string; glyph: IconName } {
  const text = (key: string) => (typeof args[key] === "string" ? (args[key] as string) : "");
  switch (name) {
    case "read_file":
      return { verb: s.toolRead, target: text("path"), glyph: "file" };
    case "list_files":
      return { verb: s.toolList, target: text("path") || ".", glyph: "folder" };
    case "glob_files":
      return { verb: s.toolGlob, target: text("pattern"), glyph: "folder" };
    case "grep_files":
      return { verb: s.toolSearch, target: text("pattern"), glyph: "search" };
    case "write_file":
      return { verb: s.toolWrite, target: text("path"), glyph: "pencil" };
    case "edit_file":
      return { verb: s.toolEdit, target: text("path"), glyph: "pencil" };
    case "terminal_run":
      return { verb: s.toolRun, target: text("command"), glyph: "terminal" };
    case "terminal_read":
    case "terminal_list":
    case "terminal_close":
      return { verb: s.toolTerminal, target: name.slice("terminal_".length), glyph: "terminal" };
    case "current_transcript":
      return { verb: s.toolTranscript, target: "", glyph: "waveform" };
    case "current_minutes":
      return { verb: s.toolMinutes, target: "", glyph: "minutes" };
    default:
      return { verb: name, target: describeArgs(args), glyph: "tool" };
  }
}

function describeArgs(args: Record<string, unknown>, limit = 90): string {
  const entries = Object.entries(args);
  const text =
    entries.length === 1 && typeof entries[0][1] === "string"
      ? (entries[0][1] as string)
      : entries.map(([key, value]) => `${key}: ${render(value)}`).join(", ");
  const flat = text.replace(/\s+/g, " ");
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1)}…`;
}

function render(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

function ringSvg(): { svg: SVGSVGElement; arc: SVGCircleElement } {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 18 18");
  svg.setAttribute("width", "18");
  svg.setAttribute("height", "18");
  svg.setAttribute("aria-hidden", "true");
  const track = document.createElementNS(SVG, "circle");
  const arc = document.createElementNS(SVG, "circle");
  for (const circle of [track, arc]) {
    circle.setAttribute("cx", "9");
    circle.setAttribute("cy", "9");
    circle.setAttribute("r", "7");
  }
  track.setAttribute("class", "ring-track");
  arc.setAttribute("class", "ring-arc");
  arc.setAttribute("stroke-dasharray", String(RING));
  arc.setAttribute("transform", "rotate(-90 9 9)");
  svg.append(track, arc);
  return { svg, arc };
}

function elapsed(ms: number, whole = false): string {
  if (ms < 1000 && !whole) return `${Math.max(0, ms).toFixed(0)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return whole ? `${Math.floor(seconds)}s` : `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
}

export function compactNumber(value: number): string {
  if (value < 1000) return String(Math.round(value));
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k`;
  return `${(value / 1_000_000).toFixed(1)}M`;
}
