/**
 * The chat panel.
 *
 * Driven by the websocket rather than the REST endpoint, because a turn that
 * calls tools is exactly the case where waiting is longest and the intermediate
 * steps are the most interesting thing on screen. A model reading three files
 * should look like a model reading three files, not like a spinner that
 * resolves into a paragraph four seconds later.
 *
 * **Tool calls are rendered as structure, not prose.** The server sends the
 * name, the arguments and the result separately, so the panel shows a
 * collapsible row per call with its outcome and duration. Flattening that into
 * the message text would throw away the one thing that makes an agent
 * legible — what it actually did, as opposed to what it says it did.
 *
 * **A failed tool is shown as failed.** Red, with its error code. The
 * alternative is a transcript where the model quietly recovers from something
 * and the reader never learns anything went wrong.
 */

import { copyText, h } from "../dom";
import type { Strings } from "../i18n";
import type { Notifications } from "../toast";

interface ToolCallView {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  row: HTMLElement;
  body: HTMLElement;
}

export class ChatPanel {
  private socket: WebSocket | null = null;
  private readonly log: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly send: HTMLButtonElement;
  private readonly status: HTMLElement;
  private busy = false;
  private turn: HTMLElement | null = null;
  private readonly calls = new Map<string, ToolCallView>();
  private adapter = "";

  constructor(
    private readonly host: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
  ) {
    this.log = h("div", { class: "chat-log", role: "log", "aria-live": "polite" });
    this.status = h("div", { class: "chat-status note" });

    this.input = h("textarea", {
      class: "chat-input",
      rows: "1",
      placeholder: strings.chatPlaceholder,
      "aria-label": strings.chatPlaceholder,
    }) as HTMLTextAreaElement;

    this.send = h("button", {
      class: "btn",
      type: "button",
      text: strings.chatSend,
    }) as HTMLButtonElement;

    this.bind();
    this.host.append(
      this.log,
      h("div", { class: "chat-compose" }, this.input, this.send),
      this.status,
    );
    this.renderEmpty();
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.input.placeholder = strings.chatPlaceholder;
    this.send.textContent = strings.chatSend;
    if (this.log.querySelector(".empty")) this.renderEmpty();
  }

  private bind(): void {
    this.send.addEventListener("click", () => this.submit());
    this.input.addEventListener("keydown", (event) => {
      // Enter sends, Shift+Enter breaks the line. The opposite convention
      // exists, but every chat surface people already use works this way and
      // a surprising Enter costs a message.
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submit();
      }
    });
    // Grow with the content, to a point: a composer that scrolls at three
    // lines makes reviewing a long question impossible.
    this.input.addEventListener("input", () => {
      this.input.style.height = "auto";
      this.input.style.height = `${Math.min(180, this.input.scrollHeight)}px`;
    });
  }

  focus(): void {
    this.input.focus();
  }

  private renderEmpty(): void {
    const s = this.strings;
    this.log.replaceChildren(
      h(
        "div",
        { class: "empty" },
        h("span", { class: "k", text: "話" }),
        s.chatEmpty,
      ),
    );
  }

  /* ---------------------------------------------------------------- socket */

  private connect(): WebSocket {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) return this.socket;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/v1/chat/stream`);
    socket.addEventListener("message", (event) => this.onMessage(event));
    socket.addEventListener("close", () => {
      if (this.busy) this.finish(this.strings.cannotConnect, true);
      this.socket = null;
    });
    socket.addEventListener("error", () => {
      if (this.busy) this.finish(this.strings.cannotConnect, true);
    });
    this.socket = socket;
    return socket;
  }

  private submit(): void {
    const text = this.input.value.trim();
    if (!text || this.busy) return;

    this.input.value = "";
    this.input.style.height = "auto";
    if (this.log.querySelector(".empty")) this.log.replaceChildren();

    this.append("user", text);
    this.busy = true;
    this.send.disabled = true;
    this.status.textContent = this.strings.chatThinking;
    this.turn = null;
    this.calls.clear();

    const socket = this.connect();
    const payload = JSON.stringify({ message: text });
    if (socket.readyState === WebSocket.OPEN) socket.send(payload);
    else socket.addEventListener("open", () => socket.send(payload), { once: true });
  }

  private onMessage(event: MessageEvent): void {
    const frame = JSON.parse(event.data as string) as Record<string, unknown>;
    const type = String(frame.type ?? "");

    switch (type) {
      case "conversation":
        this.adapter = String(frame.adapter ?? "");
        break;

      case "tool/call":
        this.showCall(frame);
        break;

      case "tool/result":
        this.showResult(frame);
        break;

      case "assistant/text":
        this.showAnswer(String(frame.text ?? ""));
        break;

      case "turn/end":
        this.finish(this.summarize(frame), false);
        break;

      case "turn/error":
      case "error":
        this.notify.error(String(frame.message ?? "error"));
        this.finish("", true);
        break;
    }
  }

  private summarize(frame: Record<string, unknown>): string {
    const s = this.strings;
    const parts = [`${Number(frame.steps ?? 0)} ${s.chatSteps}`];
    const calls = Number(frame.tool_calls ?? 0);
    if (calls > 0) parts.push(`${calls} ${calls === 1 ? s.chatToolCall : s.chatToolCalls}`);
    const ms = Number(frame.duration_ms ?? 0);
    parts.push(ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`);
    if (this.adapter) parts.push(this.adapter);
    return parts.join("  ·  ");
  }

  private finish(status: string, failed: boolean): void {
    this.busy = false;
    this.send.disabled = false;
    this.status.textContent = status;
    this.status.classList.toggle("bad", failed);
    this.turn = null;
    this.input.focus();
  }

  /* ---------------------------------------------------------------- render */

  private bubble(): HTMLElement {
    if (!this.turn) {
      this.turn = h("div", { class: "msg assistant" });
      this.log.append(this.turn);
    }
    return this.turn;
  }

  private append(role: "user" | "assistant", text: string): HTMLElement {
    const node = h("div", { class: `msg ${role}` });
    node.append(h("div", { class: "msg-body ja", text }));
    this.log.append(node);
    this.scroll();
    return node;
  }

  private showCall(frame: Record<string, unknown>): void {
    const id = String(frame.id ?? "");
    const name = String(frame.name ?? "");
    const args = (frame.arguments ?? {}) as Record<string, unknown>;

    const row = h("details", { class: "toolcall running" });
    const summary = h("summary");
    summary.append(
      h("span", { class: "tool-name", text: name }),
      h("span", { class: "tool-args", text: compact(args) }),
      h("span", { class: "tool-state", text: this.strings.chatRunning }),
    );
    const body = h("pre", { class: "tool-output" });
    row.append(summary, body);
    this.bubble().append(row);
    this.calls.set(id, { id, name, arguments: args, row, body });
    this.scroll();
  }

  private showResult(frame: Record<string, unknown>): void {
    const view = this.calls.get(String(frame.call_id ?? ""));
    if (!view) return;

    const ok = Boolean(frame.ok);
    const ms = Number(frame.duration_ms ?? 0);
    view.row.classList.remove("running");
    view.row.classList.toggle("failed", !ok);

    const state = view.row.querySelector(".tool-state");
    if (state) {
      // The error code, not just "failed": `no_tool` and `denied` send a
      // reader to completely different places.
      state.textContent = ok
        ? `${ms.toFixed(0)}ms`
        : `${String(frame.error ?? "error")} · ${ms.toFixed(0)}ms`;
    }
    view.body.textContent = String(frame.content ?? "");
    // A failed call is opened automatically: it is the thing the reader needs.
    if (!ok) view.row.setAttribute("open", "");
    this.scroll();
  }

  private showAnswer(text: string): void {
    const bubble = this.bubble();
    const existing = bubble.querySelector(".msg-body");
    if (existing) existing.textContent = text;
    else bubble.append(h("div", { class: "msg-body ja", text }));

    if (!bubble.querySelector(".msg-copy")) {
      const copy = h("button", {
        class: "icon-btn msg-copy",
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
      bubble.append(copy);
    }
    this.scroll();
  }

  private scroll(): void {
    this.log.scrollTop = this.log.scrollHeight;
  }
}

/** A call's arguments on one line, for the collapsed summary. */
function compact(args: Record<string, unknown>, limit = 90): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  const text = entries.map(([key, value]) => `${key}=${render(value)}`).join(" ");
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
