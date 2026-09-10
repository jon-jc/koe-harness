/**
 * The agent panel.
 *
 * Shaped like Claude Code's transcript rather than like a messaging app, and
 * the reason is what an agent's output *is*. A chat app is built for two
 * people taking turns; bubbles, alignment and avatars all say "who spoke".
 * An agent turn is mostly not speech — it is a sequence of actions with a
 * little prose between them — and the question a reader has is "what did it
 * do, in what order, and did it work". So:
 *
 *   - **One column, one marker per entry.** `>` is you, `⏺` is the agent.
 *     A tool call is a `⏺` whose colour is its state — amber while running,
 *     green when it returned, red when it failed — so a failed call is
 *     findable by colour from across the room.
 *   - **Results hang under their call** with `⎿`, previewed to three lines.
 *     The full output is one click away; forty lines of `ls` inline would bury
 *     the answer it was gathered for.
 *   - **A working line, not a spinner.** Elapsed time and step number, because
 *     a turn doing six model calls and a turn waiting on one slow one look
 *     identical behind a spinner, and they call for different patience.
 *   - **The composer steers.** While a turn runs, what you type joins it at
 *     the next step boundary instead of waiting behind it — the harness inbox,
 *     made reachable. Esc interrupts, as it does in every terminal agent.
 *
 * The panel owns the socket and nothing else. What the app shows elsewhere —
 * the context gauge in the status bar, the harness sidebar — it learns
 * through `ChatEvents`, so the panel never reaches into the page around it.
 */

import { copyText, h } from "../dom";
import type { Strings } from "../i18n";
import { renderMarkdown } from "../markdown";
import type { AgentActivity, SlashCommand } from "../state";
import { IDLE_AGENT } from "../state";
import type { Notifications } from "../toast";

export interface ContextGauge {
  total: number;
  window: number;
  pressure: number;
}

export interface ChatEvents {
  onContext?: (gauge: ContextGauge) => void;
  onActivity?: (activity: AgentActivity) => void;
  onCommands?: (commands: readonly SlashCommand[]) => void;
}

interface ToolView {
  row: HTMLElement;
  meta: HTMLElement;
  result: HTMLElement;
  name: string;
}

/** Claude Code's glyph cycle. Ping-ponged, so it breathes rather than ticks. */
const SPINNER = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];
const PREVIEW_LINES = 3;
const HISTORY_LIMIT = 50;

export class ChatPanel {
  private socket: WebSocket | null = null;
  private readonly log: HTMLElement;
  private readonly working: HTMLElement;
  private readonly glyph: HTMLElement;
  private readonly workingText: HTMLElement;
  private readonly workingMeta: HTMLElement;
  private readonly box: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly menu: HTMLElement;
  private readonly hintLeft: HTMLElement;
  private readonly hintRight: HTMLElement;

  private activity: AgentActivity = IDLE_AGENT;
  private turnStarted = 0;
  private ticker = 0;
  private frame = 0;
  private runningTool = "";
  private readonly calls = new Map<string, ToolView>();

  private commands: readonly SlashCommand[] = [];
  private menuItems: SlashCommand[] = [];
  private menuActive = 0;

  private readonly history: string[] = [];
  private historyAt = -1;
  private readonly still = window.matchMedia("(prefers-reduced-motion: reduce)");

  constructor(
    private readonly host: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
    private readonly events: ChatEvents = {},
  ) {
    this.log = h("div", { class: "cc-log", role: "log", "aria-live": "polite" });

    this.glyph = h("span", { class: "cc-glyph", "aria-hidden": "true", text: "✻" });
    this.workingText = h("span", { class: "cc-working-text" });
    this.workingMeta = h("span", { class: "cc-working-meta" });
    this.working = h("div", { class: "cc-working", role: "status" }, this.glyph, this.workingText, this.workingMeta);
    this.working.hidden = true;

    this.input = h("textarea", {
      class: "cc-input",
      rows: "1",
      spellcheck: "false",
      "aria-label": strings.chatPlaceholder,
      placeholder: strings.chatPlaceholder,
    });
    this.menu = h("div", { class: "cc-menu", role: "listbox" });
    this.menu.hidden = true;
    this.box = h(
      "div",
      { class: "cc-box" },
      h("span", { class: "cc-prompt", "aria-hidden": "true", text: ">" }),
      this.input,
    );
    this.hintLeft = h("span", { class: "cc-hint" });
    this.hintRight = h("span", { class: "cc-hint mono" });

    this.host.append(
      this.log,
      this.working,
      h(
        "div",
        { class: "cc-compose" },
        this.menu,
        this.box,
        h("div", { class: "cc-hints" }, this.hintLeft, h("span", { class: "grow" }), this.hintRight),
      ),
    );

    this.bind();
    this.renderWelcome();
    this.renderHints();
    // Opened at once: the server answers the command list before any
    // conversation exists, so completion works for the first thing typed.
    this.connect();
  }

  /* ---------------------------------------------------------------- public */

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.input.setAttribute("aria-label", strings.chatPlaceholder);
    this.renderHints();
    if (this.log.querySelector(".cc-welcome")) this.renderWelcome();
    if (this.activity.busy) this.tick();
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
    this.input.value = `${quoted}\n\n${current}`;
    this.grow();
    this.input.focus();
    this.input.setSelectionRange(this.input.value.length, this.input.value.length);
  }

  /** Put text in the composer without sending it. */
  insert(text: string): void {
    this.input.value = text;
    this.grow();
    this.updateMenu();
    this.input.focus();
  }

  /**
   * Start over.
   *
   * Closing the socket is the reset: the server disposes the agent with the
   * connection, so there is no half-cleared session left behind.
   */
  clear(): void {
    this.socket?.close();
    this.socket = null;
    this.calls.clear();
    this.setActivity({ ...IDLE_AGENT, model: this.activity.model });
    this.events.onContext?.({ total: 0, window: 0, pressure: 0 });
    this.renderWelcome();
    this.connect();
    this.input.focus();
  }

  /* ---------------------------------------------------------------- input */

  private bind(): void {
    this.input.addEventListener("keydown", (event) => this.onKey(event));
    this.input.addEventListener("input", () => {
      this.historyAt = -1;
      this.grow();
      this.updateMenu();
    });
    this.box.addEventListener("click", () => this.input.focus());
    this.menu.addEventListener("mousedown", (event) => {
      // mousedown, not click: a click would blur the input first and close
      // the menu out from under the pointer.
      const item = (event.target as HTMLElement).closest<HTMLElement>("[data-command]");
      if (!item) return;
      event.preventDefault();
      this.runCommand(item.dataset.command ?? "");
    });
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
    if (menuOpen && event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      this.runCommand(this.menuItems[this.menuActive].name);
      return;
    }

    if (event.key === "Escape") {
      if (menuOpen) {
        this.menu.hidden = true;
      } else if (this.activity.busy) {
        this.cancel();
      } else {
        return;
      }
      // Handled here, so the page's own Escape (clear a selection, close
      // search) does not also fire behind it.
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      // isComposing: Enter confirms a kana-to-kanji conversion, and sending
      // the half-converted sentence is the single most irritating thing a
      // chat box can do to someone typing Japanese.
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
      return;
    }
    if (event.key === "ArrowDown" && this.historyAt >= 0 && this.caretOnLastLine()) {
      event.preventDefault();
      this.historyAt -= 1;
      this.input.value =
        this.historyAt >= 0 ? this.history[this.history.length - 1 - this.historyAt] : "";
      this.grow();
    }
  }

  private caretOnFirstLine(): boolean {
    return !this.input.value.slice(0, this.input.selectionStart).includes("\n");
  }

  private caretOnLastLine(): boolean {
    return !this.input.value.slice(this.input.selectionEnd).includes("\n");
  }

  /** Grow with the content, to a point. */
  private grow(): void {
    this.input.style.height = "auto";
    this.input.style.height = `${Math.min(220, this.input.scrollHeight)}px`;
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
      ...this.menuItems.map((command, index) => {
        const item = h(
          "div",
          {
            class: `cc-menu-item${index === this.menuActive ? " on" : ""}`,
            role: "option",
            "aria-selected": String(index === this.menuActive),
            "data-command": command.name,
          },
          h("span", { class: "cc-menu-name", text: `/${command.name}` }),
          h("span", { class: "cc-menu-summary", text: command.summary }),
        );
        return item;
      }),
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
      if (this.activity.busy) this.fail(this.strings.cannotConnect);
      this.socket = null;
    });
    this.socket = socket;
    return socket;
  }

  private send(payload: Record<string, unknown>): void {
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
    this.log.querySelector(".cc-welcome")?.remove();
    if (this.history[this.history.length - 1] !== text) this.history.push(text);
    if (this.history.length > HISTORY_LIMIT) this.history.shift();
    this.historyAt = -1;

    // While a turn is running the composer steers it rather than queueing a
    // new one: a model three tool calls into the wrong file needs telling
    // now, and cancelling to re-prompt throws away the work already done.
    const steering = this.activity.busy;
    const isCommand = /^\/[a-zA-Z][\w-]*(\s|$)/.test(text);
    this.entry(steering ? "steer" : "user", steering ? "↳" : ">").body.append(
      ...(steering ? [h("span", { class: "cc-tag", text: this.strings.chatSteer })] : []),
      h("span", { class: "cc-user-text ja", text }),
    );
    this.scroll();

    if (!steering && !isCommand) this.beginTurn();
    this.send(steering ? { steer: text } : { message: text });
  }

  private cancel(): void {
    if (!this.activity.busy) return;
    this.send({ cancel: true });
    this.workingText.textContent = this.strings.chatStopping;
  }

  /* ---------------------------------------------------------------- frames */

  private onMessage(event: MessageEvent): void {
    const frame = JSON.parse(String(event.data)) as Record<string, unknown>;
    switch (String(frame.type ?? "")) {
      case "conversation":
        this.setActivity({ ...this.activity, model: String(frame.adapter ?? "") });
        break;

      case "commands":
        this.commands = (frame.commands as SlashCommand[]) ?? [];
        this.events.onCommands?.(this.commands);
        this.updateMenu();
        break;

      case "step/start":
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
        // :last-of-type matches element type, not class, so it would miss a
        // steer followed by any other entry. Take the last one explicitly.
        const steers = this.log.querySelectorAll(".cc-steer");
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
        this.result(this.entry("notice", "⎿"), this.strings.chatInterrupted, "bad");
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
    this.turnStarted = performance.now();
    this.setActivity({ ...this.activity, busy: true, step: 0 });
    this.working.hidden = false;
    window.clearInterval(this.ticker);
    this.ticker = window.setInterval(() => this.tick(), 120);
    this.tick();
  }

  private endTurn(frame: Record<string, unknown>): void {
    const s = this.strings;
    const parts = [s.chatWorkedFor.replace("{t}", elapsed(Number(frame.duration_ms ?? 0)))];
    const steps = Number(frame.steps ?? 0);
    if (steps > 1) parts.push(`${steps} ${s.chatSteps}`);
    const calls = Number(frame.tool_calls ?? 0);
    if (calls > 0) parts.push(`${calls} ${calls === 1 ? s.chatToolCall : s.chatToolCalls}`);
    const tokens = Number(frame.input_tokens ?? 0) + Number(frame.output_tokens ?? 0);
    if (tokens > 0) parts.push(`${compactNumber(tokens)} tokens`);
    // Only when it is not the ordinary one: "completed" on every turn is
    // noise, "max-steps" on one of them is the answer to why it stopped.
    const reason = String(frame.reason ?? "");
    if (reason && reason !== "completed" && reason !== "cancelled") parts.push(reason);

    this.log.append(h("div", { class: "cc-summary" }, h("span", { text: "✻" }), parts.join(" · ")));
    this.gauge(frame.context);
    this.stopWorking();
    this.setActivity({ ...this.activity, turns: this.activity.turns + 1 });
    this.scroll();
  }

  private fail(message: string): void {
    const row = this.entry("error", "⏺");
    row.body.append(h("span", { class: "ja", text: message }));
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
    if (!this.still.matches) {
      this.frame = (this.frame + 1) % SPINNER.length;
      this.glyph.textContent = SPINNER[this.frame];
    }
    this.workingText.textContent = this.runningTool
      ? `${s.chatRunning} ${this.runningTool}…`
      : s.chatThinking;
    const parts = [elapsed(performance.now() - this.turnStarted, true)];
    if (this.activity.step > 0) parts.push(`${s.chatStep} ${this.activity.step}`);
    parts.push(s.chatEscInterrupt);
    this.workingMeta.textContent = `(${parts.join(" · ")})`;
  }

  private setActivity(activity: AgentActivity): void {
    const wasBusy = this.activity.busy;
    this.activity = activity;
    if (wasBusy !== activity.busy) this.renderHints();
    this.events.onActivity?.(activity);
  }

  private gauge(context: unknown): void {
    if (!context || typeof context !== "object") return;
    const data = context as Record<string, unknown>;
    this.events.onContext?.({
      total: Number(data.total ?? 0),
      window: Number(data.context_window ?? 0),
      pressure: Number(data.pressure ?? 0),
    });
  }

  /* ---------------------------------------------------------------- render */

  private renderHints(): void {
    const s = this.strings;
    this.hintLeft.textContent = this.activity.busy ? s.chatHintBusy : s.chatHintIdle;
    this.hintLeft.classList.toggle("steer", this.activity.busy);
    this.box.classList.toggle("steer", this.activity.busy);
    this.input.placeholder = this.activity.busy ? s.chatSteerPlaceholder : s.chatPlaceholder;
    this.hintRight.textContent = "";
  }

  private renderWelcome(): void {
    const s = this.strings;
    const suggestions = h("div", { class: "cc-suggestions" });
    for (const text of [s.chatSuggest1, s.chatSuggest2, s.chatSuggest3]) {
      const button = h("button", { class: "cc-suggestion ja", type: "button", text });
      button.addEventListener("click", () => {
        this.input.value = text;
        this.submit();
      });
      suggestions.append(button);
    }

    this.log.replaceChildren(
      h(
        "div",
        { class: "cc-welcome" },
        h(
          "div",
          { class: "cc-welcome-card" },
          h(
            "div",
            { class: "cc-welcome-title" },
            h("span", { class: "cc-welcome-mark", "aria-hidden": "true", text: "✻" }),
            h("span", { text: s.chatWelcome }),
          ),
          h("p", { class: "cc-welcome-body ja", text: s.chatEmpty }),
          h(
            "ul",
            { class: "cc-tips" },
            h("li", { class: "ja", text: s.chatTip1 }),
            h("li", { class: "ja", text: s.chatTip2 }),
            h("li", { class: "ja", text: s.chatTip3 }),
          ),
        ),
        suggestions,
      ),
    );
  }

  /** One transcript entry: a marker in the gutter, a body beside it. */
  private entry(kind: string, marker: string): { row: HTMLElement; body: HTMLElement } {
    const body = h("div", { class: "cc-body" });
    const row = h(
      "div",
      { class: `cc-entry cc-${kind}` },
      h("span", { class: "cc-marker", "aria-hidden": "true", text: marker }),
      body,
    );
    this.log.append(row);
    return { row, body };
  }

  /** The `⎿` line under an entry. */
  private result(target: { body: HTMLElement }, text: string, tone = ""): HTMLElement {
    const node = h(
      "div",
      { class: `cc-result${tone ? ` ${tone}` : ""}` },
      h("span", { class: "cc-elbow", "aria-hidden": "true", text: "⎿" }),
      h("span", { class: "cc-result-text", text }),
    );
    target.body.append(node);
    return node;
  }

  private showAnswer(text: string): void {
    if (!text.trim()) return;
    const { row, body } = this.entry("assistant", "⏺");
    body.append(renderMarkdown(text));

    const copy = h("button", {
      class: "icon-btn sm cc-copy",
      type: "button",
      title: this.strings.copy,
      "aria-label": this.strings.copy,
      text: "⧉",
    });
    copy.addEventListener("click", () => {
      void copyText(text).then((ok) =>
        ok ? this.notify.ok(this.strings.copied) : this.notify.error(this.strings.copyFailed),
      );
    });
    row.append(copy);
    this.scroll();
  }

  private showCall(frame: Record<string, unknown>): void {
    const id = String(frame.id ?? "");
    const name = String(frame.name ?? "");
    const args = (frame.arguments ?? {}) as Record<string, unknown>;

    const entry = this.entry("tool", "⏺");
    entry.row.classList.add("running");
    const meta = h("span", { class: "cc-tool-meta" });
    entry.body.append(
      h(
        "div",
        { class: "cc-tool-head" },
        h("span", { class: "cc-tool-name", text: name }),
        h("span", { class: "cc-tool-args", text: `(${describeArgs(args)})` }),
        meta,
      ),
    );
    const result = h("div", { class: "cc-tool-result" });
    entry.body.append(result);

    this.calls.set(id, { row: entry.row, meta, result, name });
    this.runningTool = name;
    this.setActivity({ ...this.activity, toolCalls: this.activity.toolCalls + 1 });
    this.tick();
    this.scroll();
  }

  private showResult(frame: Record<string, unknown>): void {
    // The harness keys a result by `id`, the call it answers. `call_id` is the
    // pre-harness loop's name for it, still accepted so an older server works.
    const view = this.calls.get(String(frame.id ?? frame.call_id ?? ""));
    if (!view) return;

    const ok = Boolean(frame.ok);
    const ms = Number(frame.duration_ms ?? 0);
    view.row.classList.remove("running");
    view.row.classList.add(ok ? "ok" : "failed");
    // The error code, not just "failed": `no_tool` and `denied` send a reader
    // to completely different places.
    view.meta.textContent = ok ? elapsed(ms) : `${String(frame.error ?? "error")} · ${elapsed(ms)}`;

    const content = String(frame.content ?? "").replace(/\s+$/, "");
    const lines = content ? content.split("\n") : [];
    const holder = { body: view.result };

    if (lines.length === 0) {
      this.result(holder, this.strings.chatNoOutput, "dim");
    } else if (!ok || lines.length <= PREVIEW_LINES) {
      // A failure is shown whole: it is the thing the reader needs.
      this.result(holder, content, ok ? "" : "bad");
    } else {
      const line = this.result(holder, lines.slice(0, PREVIEW_LINES).join("\n"));
      const more = h("button", {
        class: "cc-more",
        type: "button",
        text: this.strings.chatMoreLines.replace("{n}", String(lines.length - PREVIEW_LINES)),
      });
      let open = false;
      more.addEventListener("click", () => {
        open = !open;
        const text = line.querySelector(".cc-result-text");
        if (text) text.textContent = open ? content : lines.slice(0, PREVIEW_LINES).join("\n");
        more.textContent = open
          ? this.strings.chatCollapse
          : this.strings.chatMoreLines.replace("{n}", String(lines.length - PREVIEW_LINES));
      });
      line.append(more);
    }

    if (this.runningTool === view.name) this.runningTool = "";
    this.tick();
    this.scroll();
  }

  private showCompaction(frame: Record<string, unknown>): void {
    const error = frame.error ? String(frame.error) : "";
    const shadowed = Number(frame.shadowed ?? 0);
    const tokens = Number(frame.shadowed_tokens ?? 0);
    const detail = error || [
      String(frame.kind ?? ""),
      shadowed > 0 ? `${shadowed} events` : "",
      tokens > 0 ? `${compactNumber(tokens)} tokens` : "",
    ].filter(Boolean).join(" · ");

    const row = this.entry("notice", "✻");
    row.body.append(h("span", { class: "cc-notice-text", text: this.strings.chatCompacted }));
    if (detail) this.result(row, detail, error ? "bad" : "dim");
    if (!error) {
      this.setActivity({ ...this.activity, compactions: this.activity.compactions + 1 });
    }
    this.scroll();
  }

  /**
   * A command's answer.
   *
   * Rendered as the harness replying, under the line that invoked it, rather
   * than as the assistant: a transcript in which the model answered /context
   * is a transcript of something that did not happen.
   */
  private showCommand(frame: Record<string, unknown>): void {
    this.gauge(frame.context);
    if (frame.reload) {
      this.calls.clear();
      this.setActivity({ ...IDLE_AGENT, model: this.activity.model });
      this.renderWelcome();
      return;
    }
    const users = this.log.querySelectorAll<HTMLElement>(".cc-user .cc-body");
    const last = users[users.length - 1];
    const host = last ? { body: last } : this.entry("notice", "⎿");
    this.result(host, String(frame.text ?? ""), frame.ok ? "mono" : "bad mono");
    this.scroll();
  }

  private scroll(): void {
    // Follow the output only if the reader is already at the bottom. Yanking
    // someone back down while they scroll up to re-read a tool result is the
    // classic way a live log becomes unreadable.
    const distance = this.log.scrollHeight - this.log.scrollTop - this.log.clientHeight;
    if (distance < 120) this.log.scrollTop = this.log.scrollHeight;
  }
}

/**
 * A call's arguments the way Claude Code shows them: a lone string argument is
 * just its value — `read(src/app.py)` — and anything else is `key: value`.
 */
function describeArgs(args: Record<string, unknown>, limit = 72): string {
  const entries = Object.entries(args);
  let text: string;
  if (entries.length === 1 && typeof entries[0][1] === "string") {
    text = entries[0][1] as string;
  } else {
    text = entries.map(([key, value]) => `${key}: ${render(value)}`).join(", ");
  }
  text = text.replace(/\s+/g, " ");
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
}

function render(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
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
