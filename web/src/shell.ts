/**
 * The frame: sessions on the left, the conversation in the middle, and a side
 * panel on the right for the things a conversation is about — the transcript,
 * the 議事録, the code, a terminal.
 *
 * The conversation is the application, so it is the only region without a
 * size of its own: it gets whatever the other two leave, which is what makes
 * "the conversation gets the room" true rather than a coincidence of
 * flex-grow values.
 *
 * **Sizes are the user's and they persist.** Someone who drags the panel wide
 * has said something about how they work, and throwing that away on reload
 * is the kind of small rudeness that makes an application feel like a page.
 *
 * **Collapse is a toggle, not a drag to zero.** Dragging a region to nothing
 * and being unable to find it again is the classic splitter failure; the
 * minimum is enforced, and hiding is a separate, reversible action.
 *
 * **Dragging is pointer capture, not mousemove on the document.** Capture
 * keeps a drag alive when the pointer crosses the terminal's canvas or leaves
 * the window — exactly where a naive implementation drops it mid-resize.
 */

export type Region = "sidebar" | "panel";

export interface Geometry {
  /** Width in pixels. */
  size: number;
  visible: boolean;
}

const REGIONS: readonly Region[] = ["sidebar", "panel"];

/** Enough that a region is usable; below this it is a sliver nobody wants. */
const MIN: Record<Region, number> = { sidebar: 220, panel: 360 };

/** Beyond this a region is eating the conversation it is supposed to serve. */
const MAX_FRACTION: Record<Region, number> = { sidebar: 0.35, panel: 0.62 };

const DEFAULTS: Record<Region, Geometry> = {
  sidebar: { size: 268, visible: true },
  // Closed at first: the panel is where the conversation goes to look at
  // something, and opening onto one presumes what that is.
  panel: { size: 520, visible: false },
};

/**
 * Below these widths a region overlays the conversation instead of sitting
 * beside it, so on a first run it starts closed rather than covering it.
 * Only a default: once someone opens it, that choice is what persists.
 */
const OVERLAY_BELOW: Record<Region, number> = { sidebar: 760, panel: 1000 };

const STORAGE_KEY = "koe.layout.v2";

const CSS_VARIABLE: Record<Region, string> = {
  sidebar: "--sidebar-w",
  panel: "--panel-w",
};

export interface ShellEvents {
  /** A region changed size or visibility; anything that draws needs to refit. */
  onResize?: (region: Region, geometry: Readonly<Geometry>) => void;
}

export class Shell {
  private readonly geometry: Record<Region, Geometry>;

  constructor(
    private readonly root: HTMLElement,
    private readonly events: ShellEvents = {},
  ) {
    this.geometry = this.restore();
    for (const region of REGIONS) this.apply(region);
    this.bindSplitters();
  }

  /* ------------------------------------------------------------ layout */

  private restore(): Record<Region, Geometry> {
    const saved = readLayout();
    const out = {} as Record<Region, Geometry>;
    for (const region of REGIONS) {
      const entry = saved[region];
      out[region] = {
        size: Number(entry?.size ?? DEFAULTS[region].size),
        // Below the overlay width a region starts closed whatever was saved:
        // that choice was made where the region sat beside the conversation,
        // and honouring it here covered the conversation on load.
        visible: this.overlaid(region)
          ? false
          : typeof entry?.visible === "boolean"
            ? entry.visible
            : DEFAULTS[region].visible,
      };
    }
    return out;
  }

  private persist(): void {
    // Opening or closing an overlay is transient. Saving it would make a
    // drawer closed on a phone-width window stay closed on the desktop.
    const saved = readLayout();
    const out = {} as Record<Region, Geometry>;
    for (const region of REGIONS) {
      out[region] = {
        size: this.geometry[region].size,
        visible: this.overlaid(region)
          ? (saved[region]?.visible ?? DEFAULTS[region].visible)
          : this.geometry[region].visible,
      };
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
    } catch {
      // Private browsing, or site data blocked: the layout still works, it
      // just starts from the defaults next time.
    }
  }

  private clamp(region: Region, size: number): number {
    if (!Number.isFinite(size)) return DEFAULTS[region].size;
    const max = Math.max(MIN[region], window.innerWidth * MAX_FRACTION[region]);
    return Math.round(Math.min(max, Math.max(MIN[region], size)));
  }

  /** Push one region's state into the DOM. */
  private apply(region: Region): void {
    // Clamped here rather than when stored, so a window that shrinks and grows
    // back gets the size someone chose, not the one the small window forced.
    const visible = this.geometry[region].visible;
    const size = this.clamp(region, this.geometry[region].size);
    // Zero when hidden, written inline. A `.no-panel { --panel-w: 0 }` rule
    // cannot do it: an inline custom property beats any stylesheet rule, so a
    // closed panel kept reserving its full width beside the conversation.
    this.root.style.setProperty(CSS_VARIABLE[region], `${visible ? size : 0}px`);
    this.root.classList.toggle(`no-${region}`, !visible);

    const element = document.getElementById(region);
    if (element) element.hidden = !visible;
    const splitter = document.getElementById(`split-${region}`);
    if (splitter) splitter.hidden = !visible;

    this.events.onResize?.(region, { size, visible });
  }

  size(region: Region, size: number, persist = true): void {
    this.geometry[region].size = this.clamp(region, size);
    this.apply(region);
    if (persist) this.persist();
  }

  visible(region: Region): boolean {
    return this.geometry[region].visible;
  }

  /** Whether the region currently covers the conversation rather than sitting beside it. */
  overlaid(region: Region): boolean {
    return window.innerWidth < OVERLAY_BELOW[region];
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
    for (const region of REGIONS) {
      const splitter = document.getElementById(`split-${region}`);
      if (!splitter) continue;
      // The panel is on the right, so the same drag means the opposite to it.
      const direction = region === "panel" ? -1 : 1;

      splitter.addEventListener("pointerdown", (event) => {
        const pointer = event as PointerEvent;
        if (pointer.button !== 0) return;
        pointer.preventDefault();
        splitter.setPointerCapture(pointer.pointerId);
        splitter.classList.add("dragging");
        document.body.classList.add("resizing");

        const startX = pointer.clientX;
        const startSize = this.geometry[region].size;

        const move = (moveEvent: PointerEvent) => {
          // Stored once when the drag ends, not on every pointer move.
          this.size(region, startSize + (moveEvent.clientX - startX) * direction, false);
        };
        const end = () => {
          splitter.releasePointerCapture(pointer.pointerId);
          this.persist();
          splitter.classList.remove("dragging");
          document.body.classList.remove("resizing");
          splitter.removeEventListener("pointermove", move);
          splitter.removeEventListener("pointerup", end);
          splitter.removeEventListener("pointercancel", end);
        };

        splitter.addEventListener("pointermove", move);
        splitter.addEventListener("pointerup", end);
        splitter.addEventListener("pointercancel", end);
      });

      // Double-click restores the shipped size: the standard escape from a
      // layout someone has dragged into a corner.
      splitter.addEventListener("dblclick", () => this.reset(region));

      // A splitter nobody can reach with a keyboard is a control only some
      // people have.
      splitter.addEventListener("keydown", (event) => {
        const key = (event as KeyboardEvent).key;
        const step = ((event as KeyboardEvent).shiftKey ? 48 : 12) * direction;
        if (key === "ArrowRight") this.size(region, this.geometry[region].size + step);
        else if (key === "ArrowLeft") this.size(region, this.geometry[region].size - step);
        else if (key === "Enter" || key === " ") this.toggle(region);
        else return;
        event.preventDefault();
      });
    }

    // A window that shrank can leave a region wider than the space it is in,
    // and one that crossed an overlay width changes what a region is.
    let width = window.innerWidth;
    window.addEventListener("resize", () => {
      const now = window.innerWidth;
      for (const region of REGIONS) {
        const threshold = OVERLAY_BELOW[region];
        if (width >= threshold && now < threshold) {
          // Narrowed into an overlay: close it, or it covers the conversation.
          this.geometry[region].visible = false;
        } else if (width < threshold && now >= threshold) {
          // Widened out of one: back to whatever was chosen at this width.
          this.geometry[region].visible = readLayout()[region]?.visible ?? DEFAULTS[region].visible;
        }
        this.apply(region);
      }
      width = now;
    });
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
