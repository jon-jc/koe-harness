/**
 * Sessions, kept the way Claude Code keeps them: several conversations side by
 * side, each with its own agent, listed with what each one is doing.
 *
 * Every session is a ChatPanel with its own socket, so a long turn in one
 * keeps running while you start another. The list shows it working, and marks
 * it when it finishes while you were looking elsewhere.
 *
 * **Sessions last as long as the page.** The server's agent goes with its
 * socket, and a list that survived a reload would be a list of conversations
 * the model no longer remembers — which is worse than no list, because it
 * looks like memory.
 */

import { h } from "./dom";
import type { Strings } from "./i18n";
import { icon } from "./icons";
import { ChatPanel } from "./panels/chat";
import { IDLE_AGENT, type AgentActivity } from "./state";
import type { Notifications } from "./toast";

export interface Session {
  readonly id: string;
  readonly host: HTMLElement;
  readonly panel: ChatPanel;
  title: string;
  activity: AgentActivity;
  unread: boolean;
}

export interface SessionsEvents {
  /** The active session, or its title, changed. */
  onChange?: () => void;
  onModelClick?: () => void;
}

export class Sessions {
  private readonly items: Session[] = [];
  private activeId = "";
  private counter = 0;
  private model = "";

  constructor(
    private readonly container: HTMLElement,
    private readonly list: HTMLElement,
    private strings: Strings,
    private readonly notify: Notifications,
    private readonly events: SessionsEvents = {},
  ) {
    this.list.addEventListener("click", (event) => {
      const target = event.target as HTMLElement;
      const close = target.closest<HTMLElement>("[data-close]");
      if (close) {
        this.close(close.dataset.close ?? "");
        return;
      }
      const item = target.closest<HTMLElement>("[data-session]");
      if (item) this.activate(item.dataset.session ?? "");
    });
  }

  get active(): Session | undefined {
    return this.items.find((session) => session.id === this.activeId);
  }

  /**
   * Start a session — or return to the untouched one, which is what pressing
   * "new" twice should mean. Stacking empty sessions is a list of nothing.
   */
  create(): Session {
    const blank = this.items.find((session) => !session.title && session.activity.turns === 0 && !session.activity.busy);
    if (blank) {
      this.activate(blank.id);
      blank.panel.focus();
      return blank;
    }

    this.counter += 1;
    const id = `session-${this.counter}`;
    const host = h("div", { class: "session-view", id });
    this.container.append(host);

    let session: Session | undefined;
    const panel = new ChatPanel(host, this.strings, this.notify, {
      onTitle: (title) => {
        if (!session) return;
        session.title = title;
        this.renderList();
        if (session.id === this.activeId) this.events.onChange?.();
      },
      onActivity: (activity) => {
        if (!session) return;
        const finished = session.activity.busy && !activity.busy;
        session.activity = activity;
        if (finished && session.id !== this.activeId) session.unread = true;
        this.renderList();
      },
      onModelClick: () => this.events.onModelClick?.(),
    });
    panel.setModel(this.model);

    session = { id, host, panel, title: "", activity: IDLE_AGENT, unread: false };
    this.items.unshift(session);
    this.activate(id);
    return session;
  }

  activate(id: string): void {
    const session = this.items.find((item) => item.id === id);
    if (!session) return;
    this.activeId = id;
    session.unread = false;
    for (const item of this.items) item.host.hidden = item.id !== id;
    this.renderList();
    this.events.onChange?.();
  }

  close(id: string): void {
    const index = this.items.findIndex((item) => item.id === id);
    if (index < 0) return;
    const [session] = this.items.splice(index, 1);
    session.panel.dispose();
    session.host.remove();

    if (this.items.length === 0) {
      this.create();
    } else if (this.activeId === id) {
      this.activate(this.items[Math.min(index, this.items.length - 1)].id);
    } else {
      this.renderList();
    }
  }

  /** Move between sessions, as ⌘⇧[ and ] do between tabs. */
  cycle(step: number): void {
    if (this.items.length < 2) return;
    const at = this.items.findIndex((item) => item.id === this.activeId);
    this.activate(this.items[(at + step + this.items.length) % this.items.length].id);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    for (const item of this.items) item.panel.setStrings(strings);
    this.renderList();
  }

  setModel(name: string): void {
    if (name === this.model) return;
    this.model = name;
    for (const item of this.items) item.panel.setModel(name);
  }

  private renderList(): void {
    const s = this.strings;
    this.list.replaceChildren(
      ...this.items.map((session) => {
        const on = session.id === this.activeId;
        const status = h("span", { class: "session-status", "aria-hidden": "true" });
        if (session.activity.busy) status.append(h("span", { class: "spinner sm" }));
        else if (session.unread) status.append(h("span", { class: "unread-dot" }));

        const meta = session.activity.busy
          ? s.sessionWorking
          : session.activity.turns > 0
            ? `${session.activity.turns} ${session.activity.turns === 1 ? s.sessionTurn : s.sessionTurns}`
            : s.sessionNoMessages;

        return h(
          "div",
          { class: `session-item${on ? " on" : ""}${session.activity.busy ? " busy" : ""}`, role: "listitem" },
          h(
            "button",
            {
              class: "session-main",
              type: "button",
              "data-session": session.id,
              ...(on ? { "aria-current": "true" } : {}),
            },
            status,
            h(
              "span",
              { class: "session-text" },
              h("span", { class: "session-name ja", text: session.title || s.sessionUntitled }),
              h("span", { class: "session-meta", text: meta }),
            ),
          ),
          h(
            "button",
            { class: "session-close", type: "button", "data-close": session.id, title: s.sessionClose, "aria-label": s.sessionClose },
            icon("x", 13),
          ),
        );
      }),
    );
  }
}
