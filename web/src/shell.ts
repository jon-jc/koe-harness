/**
 * The workbench: which regions exist, how big they are, and who has focus.
 *
 * The layout this replaces made chat, terminal and code *mutually exclusive
 * views*. Asking the assistant about a transcript meant navigating away from
 * the transcript, and coming back meant losing where you were. No editor is
 * built that way, and the reason is that the panels are not alternatives to
 * each other — they are views onto one thing you are working on.
 *
 * So: an activity rail chooses what the sidebar shows, the centre keeps
 * whatever you were reading, the terminal is a panel you toggle rather than
 * travel to, and the agent is always there on the right.
 *
 * **Sizes are the user's and they persist.** Someone who drags the agent panel
 * wide has said something about how they work, and throwing that away on
 * reload is the kind of small rudeness that makes an application feel like a
 * web page. Stored per region, restored before first paint.
 *
 * **Collapse is a toggle, not a drag to zero.** Dragging a panel to nothing
 * and being unable to find it again is the classic splitter failure; the
 * minimum width is enforced, and hiding is a separate, reversible action with
 * a keyboard shortcut and a status-bar button.
 *
 * **Dragging is pointer-capture, not mousemove-on-document.** Capture keeps
 * the drag working when the pointer crosses an iframe or leaves the window,
 * which is exactly when a naive implementation drops it and leaves the layout
 * stuck mid-resize.
 */

/** A resizable region. */
export type Region = "sidebar" | "agent" | "panel";

interface Geometry {
  /** Pixels. Width for the side regions, height for the bottom panel. */
  size: number;
  visible: boolean;
}

/** Enough that a panel is usable; below this it is a sliver nobody wants. */
const MIN: Record<Region, number> = { sidebar: 200, agent: 300, panel: 120 };

/** Beyond this a region is eating the workspace it is supposed to serve. */
const MAX_FRACTION: Record<Region, number> = { sidebar: 0.4, agent: 0.55, panel: 0.7 };

const DEFAULTS: Record<Region, Geometry> = {
  sidebar: { size: 264, visible: true },
  agent: { size: 400, visible: true },
  // Hidden by default: a terminal is something you reach for, and opening the
  // app with one already taking a third of the window presumes an answer.
  panel: { size: 260, visible: false },
};

/**
 * Below these widths a region overlays the centre instead of sitting beside
 * it, so on a first run it starts closed rather than covering the transcript.
 * Only a default: once someone opens it, that choice is what persists.
 */
const OVERLAY_BELOW: Record<Region, number> = { sidebar: 720, agent: 1100, panel: 0 };

const STORAGE_KEY = "koe.layout";

const CSS_VARIABLE: Record<Region, string> = {
  sidebar: "--sidebar-w",
  agent: "--agent-w",
  panel: "--panel-h",
};

export interface ShellEvents {
  /** A region changed size or visibility; panels that draw need to refit. */
  onResize?: (region: Region) => void;
  /** The sidebar's active section changed. */
  onSide?: (name: string) => void;
}

export class Shell {
  private readonly root: HTMLElement;
  private readonly geometry: Record<Region, Geometry>;
  private side = "session";

  constructor(
    root: HTMLElement,
    private readonly events: ShellEvents = {},
  ) {
    this.root = root;
    this.geometry = this.restore();
    for (const region of ["sidebar", "agent", "panel"] as Region[]) this.apply(region);
    this.bindSplitters();
    this.bindActivity();
  }

  /* ------------------------------------------------------------ layout */

  private restore(): Record<Region, Geometry> {
    const saved = readLayout();
    const out = {} as Record<Region, Geometry>;
    for (const region of ["sidebar", "agent", "panel"] as Region[]) {
      const entry = saved[region];
      out[region] = {
        size: this.clamp(region, Number(entry?.size ?? DEFAULTS[region].size)),
        visible:
          typeof entry?.visible === "boolean"
            ? entry.visible
            : DEFAULTS[region].visible && window.innerWidth >= OVERLAY_BELOW[region],
      };
    }
    return out;
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.geometry));
    } catch {
      // Private browsing, or site data blocked: the layout still works, it
      // just starts from the defaults next time.
    }
  }

  private clamp(region: Region, size: number): number {
    const extent = region === "panel" ? window.innerHeight : window.innerWidth;
    const max = Math.max(MIN[region], extent * MAX_FRACTION[region]);
    if (!Number.isFinite(size)) return DEFAULTS[region].size;
    return Math.round(Math.min(max, Math.max(MIN[region], size)));
  }

  /** Push one region's state into the DOM. */
  private apply(region: Region): void {
    // Clamped here rather than when stored, so a window that shrinks and grows
    // back gets the size someone chose, not the one the small window forced.
    const { visible } = this.geometry[region];
    const size = this.clamp(region, this.geometry[region].size);
    this.root.style.setProperty(CSS_VARIABLE[region], `${size}px`);
    this.root.classList.toggle(`no-${region}`, !visible);

    const element = document.getElementById(region);
    if (element) element.hidden = !visible;
    const splitter = document.getElementById(
      region === "panel" ? "split-panel" : `split-${region}`,
    );
    if (splitter) splitter.hidden = !visible;

    this.events.onResize?.(region);
  }

  size(region: Region, size: number, persist = true): void {
    this.geometry[region].size = this.clamp(region, size);
    this.apply(region);
    if (persist) this.persist();
  }

  visible(region: Region): boolean {
    return this.geometry[region].visible;
  }

  show(region: Region, visible = true): void {
    this.geometry[region].visible = visible;
    this.apply(region);
    this.persist();
  }

  toggle(region: Region): boolean {
    const next = !this.geometry[region].visible;
    this.show(region, next);
    return next;
  }

  /** Put a region back to its shipped size, for a layout someone has lost. */
  reset(region: Region): void {
    this.geometry[region] = { ...DEFAULTS[region], visible: this.geometry[region].visible };
    this.apply(region);
    this.persist();
  }

  /* --------------------------------------------------------- splitters */

  private bindSplitters(): void {
    const pairs: [string, Region][] = [
      ["split-sidebar", "sidebar"],
      ["split-agent", "agent"],
      ["split-panel", "panel"],
    ];

    for (const [id, region] of pairs) {
      const splitter = document.getElementById(id);
      if (!splitter) continue;

      splitter.addEventListener("pointerdown", (event) => {
        const pointer = event as PointerEvent;
        if (pointer.button !== 0) return;
        pointer.preventDefault();
        // Capture, so the drag survives the pointer crossing the terminal's
        // canvas or leaving the window — the two places a document-level
        // mousemove listener silently stops receiving events.
        splitter.setPointerCapture(pointer.pointerId);
        splitter.classList.add("dragging");
        document.body.classList.add("resizing");
        document.body.classList.toggle("rows", region === "panel");

        const startPosition = region === "panel" ? pointer.clientY : pointer.clientX;
        const startSize = this.geometry[region].size;
        // The agent panel is on the right and the sidebar on the left, so the
        // same drag direction means opposite things to them.
        const direction = region === "agent" || region === "panel" ? -1 : 1;

        const move = (moveEvent: PointerEvent) => {
          const now = region === "panel" ? moveEvent.clientY : moveEvent.clientX;
          // Stored once when the drag ends, not on every pointer move.
          this.size(region, startSize + (now - startPosition) * direction, false);
        };
        const end = () => {
          splitter.releasePointerCapture(pointer.pointerId);
          this.persist();
          splitter.classList.remove("dragging");
          document.body.classList.remove("resizing", "rows");
          splitter.removeEventListener("pointermove", move);
          splitter.removeEventListener("pointerup", end);
          splitter.removeEventListener("pointercancel", end);
        };

        splitter.addEventListener("pointermove", move);
        splitter.addEventListener("pointerup", end);
        splitter.addEventListener("pointercancel", end);
      });

      // Double-click restores the shipped size. The standard escape from a
      // layout someone has dragged into a corner.
      splitter.addEventListener("dblclick", () => this.reset(region));

      // A splitter nobody can reach with a keyboard is a control only some
      // people have.
      splitter.addEventListener("keydown", (event) => {
        const key = (event as KeyboardEvent).key;
        const step = (event as KeyboardEvent).shiftKey ? 48 : 12;
        const grow = region === "panel" ? "ArrowUp" : "ArrowRight";
        const shrink = region === "panel" ? "ArrowDown" : "ArrowLeft";
        const sign = region === "agent" ? -1 : 1;

        if (key === grow) this.size(region, this.geometry[region].size + step * sign);
        else if (key === shrink) this.size(region, this.geometry[region].size - step * sign);
        else if (key === "Enter" || key === " ") this.toggle(region);
        else return;
        event.preventDefault();
      });
    }

    // A window that shrank can leave a region wider than the space it is in.
    window.addEventListener("resize", () => {
      for (const region of ["sidebar", "agent", "panel"] as Region[]) this.apply(region);
    });
  }

  /* ---------------------------------------------------------- activity */

  private bindActivity(): void {
    const rail = document.getElementById("activity");
    rail?.addEventListener("click", (event) => {
      const button = (event.target as HTMLElement).closest<HTMLElement>("[data-side]");
      if (!button?.dataset.side) return;
      // Clicking the section you are already in collapses the sidebar, which
      // is what every editor does and what a second click on a chosen thing
      // should mean.
      if (button.dataset.side === this.side && this.visible("sidebar")) {
        this.show("sidebar", false);
        return;
      }
      this.show("sidebar", true);
      this.showSide(button.dataset.side);
    });
  }

  /** Swap the sidebar's contents. */
  showSide(name: string): void {
    this.side = name;
    for (const pane of document.querySelectorAll<HTMLElement>(".side-pane")) {
      pane.hidden = pane.id !== `side-${name}`;
    }
    for (const button of document.querySelectorAll<HTMLElement>("[data-side]")) {
      const on = button.dataset.side === name;
      button.classList.toggle("on", on);
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
    }
    this.events.onSide?.(name);
  }

  get activeSide(): string {
    return this.side;
  }
}

function readLayout(): Partial<Record<Region, Partial<Geometry>>> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Partial<Record<Region, Partial<Geometry>>>) : {};
  } catch {
    return {};
  }
}
