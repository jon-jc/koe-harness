/**
 * The terminal panel.
 *
 * Deliberately not a terminal emulator. The server already strips escape
 * sequences, so what arrives is plain text, and rendering plain text needs a
 * `<pre>` rather than the ~250 KB xterm.js would add to a 20 KB client. The
 * cost of that choice is honest and stated in the docs: no colour, and no
 * full-screen programs.
 *
 * Two behaviours worth having anyway, because their absence is what makes a
 * naive terminal box feel broken:
 *
 * **Command history on the arrow keys.** The first thing anyone does in a
 * terminal is press Up. A box that moves the caret instead is a box people
 * stop using.
 *
 * **A command that outlasts its budget keeps streaming.** The server returns
 * `timeout` when the wait ran out rather than the command finishing, and the
 * panel then polls for output until the shell goes quiet — so `npm install`
 * fills in progressively instead of appearing to hang and then dumping.
 */

import { el, h } from "../dom";
import type { Strings } from "../i18n";
import type { Notifications } from "../toast";

interface SessionInfo {
  id: string;
  name: string;
  pid: number;
  alive: boolean;
  cwd: string;
}

interface SendResult {
  output: string;
  reason: "prompt" | "inferred_idle" | "timeout" | "session_exit";
  duration_ms: number;
}

/** How often to collect output while a command is still running. */
const POLL_MS = 400;

/** Give up following after this long with nothing new. */
const QUIET_POLLS = 12;

export class TerminalPanel {
  private readonly screen: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly status: HTMLElement;
  private session: SessionInfo | null = null;
  private busy = false;
  private readonly history: string[] = [];
  private historyAt = 0;
  private following = 0;

  constructor(
    private readonly host: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
  ) {
    this.screen = h("pre", { class: "term-screen", tabindex: "0", role: "log" });
    this.status = h("div", { class: "note term-status" });
    this.input = h("input", {
      class: "term-input",
      type: "text",
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": strings.terminalPrompt,
      placeholder: strings.terminalPrompt,
    }) as HTMLInputElement;

    const prompt = h("div", { class: "term-compose" }, h("span", { class: "term-sigil", text: "$" }), this.input);
    this.host.append(this.screen, prompt, this.status);
    this.bind();
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.input.placeholder = strings.terminalPrompt;
    this.input.setAttribute("aria-label", strings.terminalPrompt);
  }

  private bind(): void {
    this.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void this.run(this.input.value);
        return;
      }
      // The first thing anyone does in a terminal is press Up.
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        if (this.history.length === 0) return;
        event.preventDefault();
        this.historyAt = Math.max(
          0,
          Math.min(this.history.length, this.historyAt + (event.key === "ArrowUp" ? -1 : 1)),
        );
        this.input.value = this.history[this.historyAt] ?? "";
        // Caret to the end, or editing a recalled command starts mid-string.
        requestAnimationFrame(() => this.input.setSelectionRange(999, 999));
      }
      if (event.key === "l" && event.ctrlKey) {
        event.preventDefault();
        this.screen.replaceChildren();
      }
    });
    // Clicking anywhere on the output focuses the prompt, the way a real
    // terminal behaves.
    this.screen.addEventListener("click", () => {
      if (window.getSelection()?.toString()) return;
      this.input.focus();
    });
  }

  focus(): void {
    this.input.focus();
    if (!this.session) void this.open();
  }

  /* ---------------------------------------------------------------- session */

  private async open(): Promise<void> {
    this.status.textContent = this.strings.terminalStarting;
    try {
      const response = await fetch("/v1/terminal/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "panel" }),
      });
      if (response.status === 503) {
        // The capability is not mounted rather than broken, and saying which
        // is the difference between "enable the plugin" and "file a bug".
        this.write(this.strings.terminalDisabled);
        this.status.textContent = "";
        this.input.disabled = true;
        return;
      }
      if (!response.ok) throw new Error(String(response.status));
      this.session = (await response.json()) as SessionInfo;
      this.status.textContent = `${this.session.cwd}  ·  pid ${this.session.pid}`;
      this.write(`${this.strings.terminalReady}\n`);
    } catch {
      this.notify.error(this.strings.cannotConnect);
      this.status.textContent = "";
    }
  }

  private async run(command: string): Promise<void> {
    const text = command.trim();
    if (!text || this.busy) return;
    if (!this.session) {
      await this.open();
      if (!this.session) return;
    }

    this.history.push(text);
    this.historyAt = this.history.length;
    this.input.value = "";
    this.busy = true;
    this.input.disabled = true;
    this.echo(text);

    try {
      const response = await fetch(`/v1/terminal/sessions/${this.session.id}/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, timeout_s: 20 }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        this.write(`${String(body.detail ?? response.status)}\n`, "bad");
        // A dead session should not leave the panel wedged: dropping it means
        // the next command opens a fresh one.
        if (response.status === 404 || response.status === 409) this.session = null;
        return;
      }
      const result = (await response.json()) as SendResult;
      if (result.output) this.write(`${result.output}\n`);

      if (result.reason === "timeout") {
        // Still running. Follow it rather than leaving the reader to guess.
        this.write(`${this.strings.terminalStillRunning}\n`, "dim");
        await this.follow();
      } else if (result.reason === "session_exit") {
        this.write(`${this.strings.terminalExited}\n`, "dim");
        this.session = null;
      }
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      this.busy = false;
      this.input.disabled = false;
      this.input.focus();
    }
  }

  /** Poll for output until the command stops producing any. */
  private async follow(): Promise<void> {
    const session = this.session;
    if (!session) return;
    const token = ++this.following;
    let quiet = 0;

    while (quiet < QUIET_POLLS && token === this.following) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      try {
        const response = await fetch(`/v1/terminal/sessions/${session.id}/output`);
        if (!response.ok) return;
        const { output } = (await response.json()) as { output: string };
        if (output) {
          this.write(output);
          quiet = 0;
        } else {
          quiet += 1;
        }
      } catch {
        return;
      }
    }
  }

  /* ---------------------------------------------------------------- render */

  private echo(command: string): void {
    const line = h("div", { class: "term-echo" });
    line.append(h("span", { class: "term-sigil", text: "$" }), h("span", { text: ` ${command}` }));
    this.screen.append(line);
    this.scroll();
  }

  private write(text: string, tone = ""): void {
    // textContent, never innerHTML: this is process output, and a build log
    // containing a `<script>` is a normal thing for a build log to contain.
    this.screen.append(h("span", { class: `term-out ${tone}`.trim(), text }));
    this.scroll();
  }

  private scroll(): void {
    this.screen.scrollTop = this.screen.scrollHeight;
  }
}

export function mountTerminal(
  id: string,
  strings: Strings,
  notify: Notifications,
): TerminalPanel {
  return new TerminalPanel(el(id), strings, notify);
}
