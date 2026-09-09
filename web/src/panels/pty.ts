/**
 * The interactive terminal: a real emulator over a real pty.
 *
 * This is the second of two terminal panels, and the split is the point. The
 * pipe panel renders plain text and is what a model's output looks like; this
 * one runs xterm.js against a pty, so `vim`, `top` and `less` draw the way
 * they would in any other terminal.
 *
 * **The emulator is a real dependency and it is worth it here.** ~250 KB
 * against a client that was 26 KB gzipped, which is why it is loaded only when
 * this panel is first opened rather than at startup — someone who came to
 * record a meeting never pays for it. Rendering escape sequences by hand is
 * not a smaller version of this; it is a worse terminal emulator.
 *
 * **The socket carries frames, not requests.** `vim` redraws on every
 * keystroke, so a round trip has to be a websocket frame. The polling the
 * pipe panel does is right there and wrong here.
 *
 * **The pty dies with the socket.** There is no id to reconnect with and no
 * scrollback to restore: a terminal nobody can reach is a leaked shell. The
 * pipe sessions are the ones that survive a reload.
 */

import type { FitAddon } from "@xterm/addon-fit";
import type { Terminal } from "@xterm/xterm";

import { h } from "../dom";
import type { Strings } from "../i18n";
import type { Notifications } from "../toast";


/**
 * One named export from a module that may or may not be real ESM.
 *
 * Both xterm packages ship CommonJS, and esbuild bundling CJS into ESM output
 * cannot see their named exports statically — so the chunk exposes a single
 * `default` holding the module object, and destructuring `{ Terminal }`
 * yields `undefined`. The symptom is a bare "e is not a constructor" from
 * minified code, which says nothing about the cause. Checking both shapes
 * costs a line and survives either package going ESM later.
 */
function named<T>(module: unknown, key: string): T {
  const direct = (module as Record<string, unknown>)[key];
  if (direct) return direct as T;
  const fallback = (module as { default?: Record<string, unknown> }).default?.[key];
  if (fallback) return fallback as T;
  throw new Error(`the terminal emulator has no ${key} export`);
}

/** Matches the app's own palette, so the terminal is part of the window. */
const THEME_DARK = {
  background: "#090b0e",
  foreground: "#e8ebf0",
  cursor: "#5aa9e6",
  selectionBackground: "#16283a",
  black: "#14191f",
  red: "#e8695c",
  green: "#56c271",
  yellow: "#d99b3f",
  blue: "#5aa9e6",
  magenta: "#b18ce0",
  cyan: "#4fc2cd",
  white: "#a6aebb",
  brightBlack: "#3d4754",
  brightRed: "#e8836d",
  brightGreen: "#62c98c",
  brightYellow: "#dcae54",
  brightBlue: "#7cc0f2",
  brightMagenta: "#c9a8ea",
  brightCyan: "#6fd3dc",
  brightWhite: "#e8ebf0",
};

const THEME_LIGHT = {
  ...THEME_DARK,
  background: "#f6f7f9",
  foreground: "#1c222a",
  cursor: "#1f6fb2",
  selectionBackground: "#d6e6f5",
  white: "#4a5260",
  brightWhite: "#1c222a",
};

export class PtyPanel {
  private readonly screen: HTMLElement;
  private readonly status: HTMLElement;
  private terminal: Terminal | null = null;
  private fit: FitAddon | null = null;
  private socket: WebSocket | null = null;
  private starting = false;
  private observer: ResizeObserver | null = null;

  constructor(
    private readonly host: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
  ) {
    this.screen = h("div", { class: "pty-screen" });
    this.status = h("div", { class: "note term-status" });
    this.host.append(this.screen, this.status);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
  }

  /** Repaint for a theme change; xterm holds resolved colours, not variables. */
  setTheme(dark: boolean): void {
    if (this.terminal) this.terminal.options.theme = dark ? THEME_DARK : THEME_LIGHT;
  }

  async activate(): Promise<void> {
    if (this.terminal || this.starting) {
      this.terminal?.focus();
      return;
    }
    this.starting = true;
    this.status.textContent = this.strings.terminalStarting;

    try {
      // Loaded here rather than at module scope: the import is ~280 KB, and
      // esbuild splits it into the panel's own chunk only if it is dynamic.
      const [xterm, addon] = await Promise.all([
        import("@xterm/xterm"),
        import("@xterm/addon-fit"),
      ]);
      const Terminal = named<typeof import("@xterm/xterm").Terminal>(xterm, "Terminal");
      const FitAddon = named<typeof import("@xterm/addon-fit").FitAddon>(addon, "FitAddon");

      const dark = !document.documentElement.matches('[data-theme="light"]');
      const terminal = new Terminal({
        fontFamily:
          'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
        fontSize: 12.5,
        lineHeight: 1.25,
        cursorBlink: true,
        // Deep enough to hold a build log, bounded so a runaway command does
        // not become a memory leak in the browser as well as the server.
        scrollback: 5000,
        theme: dark ? THEME_DARK : THEME_LIGHT,
        allowProposedApi: true,
      });
      const fit = new FitAddon();
      terminal.loadAddon(fit);
      terminal.open(this.screen);
      fit.fit();

      this.terminal = terminal;
      this.fit = fit;
      this.connect();
      this.watchResize();
      terminal.focus();
    } catch (error) {
      this.notify.error(this.strings.terminalEmulatorFailed);
      this.status.textContent = String(error);
    } finally {
      this.starting = false;
    }
  }

  /* ---------------------------------------------------------------- socket */

  private connect(): void {
    const terminal = this.terminal;
    const fit = this.fit;
    if (!terminal || !fit) return;

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/v1/terminal/pty`);
    this.socket = socket;

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({ cols: terminal.cols, rows: terminal.rows }));
    });

    socket.addEventListener("message", (event) => {
      const frame = JSON.parse(event.data as string) as Record<string, unknown>;
      switch (frame.t) {
        case "ready":
          this.status.textContent = `${String(frame.cwd ?? "")}  ·  pid ${String(frame.pid ?? "")}`;
          break;
        case "o":
          terminal.write(String(frame.d ?? ""));
          break;
        case "exit":
          // Written into the terminal rather than shown as a toast: it belongs
          // in the scrollback with the session it ended.
          terminal.write(`\r\n\x1b[2m${this.strings.terminalExited}\x1b[0m\r\n`);
          this.status.textContent = "";
          break;
        case "fatal":
          this.showUnavailable(String(frame.message ?? ""));
          break;
      }
    });

    socket.addEventListener("close", () => {
      this.socket = null;
    });
    socket.addEventListener("error", () => {
      this.notify.error(this.strings.cannotConnect);
    });

    // Every keystroke, including the control sequences an editor needs.
    terminal.onData((data) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ t: "i", d: data }));
      }
    });
  }

  private showUnavailable(message: string): void {
    // Written as terminal output so the remedy sits where someone is already
    // looking, rather than in a toast they have to have caught.
    this.terminal?.write(`\x1b[33m${message}\x1b[0m\r\n`);
    this.status.textContent = "";
  }

  /* ---------------------------------------------------------------- resize */

  private watchResize(): void {
    // Both halves are needed: xterm has to re-lay-out its own grid, and the
    // shell has to be told, or a program that draws a full screen wraps at
    // the old width and looks corrupted.
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(this.screen);
  }

  private resize(): void {
    if (!this.terminal || !this.fit) return;
    try {
      this.fit.fit();
    } catch {
      // Fitting a hidden element throws; the next activate() refits.
      return;
    }
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(
        JSON.stringify({ t: "r", cols: this.terminal.cols, rows: this.terminal.rows }),
      );
    }
  }

  /** Called when the panel is shown again, since it may have been resized hidden. */
  refit(): void {
    requestAnimationFrame(() => this.resize());
    this.terminal?.focus();
  }

  dispose(): void {
    this.observer?.disconnect();
    this.socket?.close();
    this.terminal?.dispose();
    this.terminal = null;
  }
}
