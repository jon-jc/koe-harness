/**
 * Icons, drawn here rather than imported.
 *
 * Two dozen 24px line glyphs is a few kilobytes of path data; an icon package
 * is a dependency, a licence notice and a build step for the same result.
 * Every glyph strokes with `currentColor`, so an icon follows the text colour
 * and the theme of whatever it sits in without a colour rule of its own.
 */

type Shape =
  | readonly ["path", string, boolean?]
  | readonly ["circle", number, number, number, boolean?]
  | readonly ["rect", number, number, number, number, number, boolean?];

const ICONS = {
  plus: [["path", "M12 5v14M5 12h14"]],
  x: [["path", "M6 6l12 12M18 6 6 18"]],
  "arrow-up": [["path", "M12 19V5M5.5 11.5 12 5l6.5 6.5"]],
  stop: [["rect", 7, 7, 10, 10, 2, true]],
  "chevron-right": [["path", "m9 6 6 6-6 6"]],
  "chevron-down": [["path", "m6 9 6 6 6-6"]],
  sidebar: [["rect", 3, 4, 18, 16, 3], ["path", "M9.5 4v16"]],
  panel: [["rect", 3, 4, 18, 16, 3], ["path", "M14.5 4v16"]],
  search: [["circle", 11, 11, 6.5], ["path", "m20 20-4.2-4.2"]],
  settings: [["path", "M4 7h8M16 7h4M4 17h3M11 17h9"], ["circle", 14, 7, 2], ["circle", 9, 17, 2]],
  help: [["circle", 12, 12, 9], ["path", "M9.6 9.3a2.5 2.5 0 1 1 3.4 2.4c-.7.3-1 .9-1 1.6v.3M12 17h.01"]],
  contrast: [["circle", 12, 12, 9], ["path", "M12 3a9 9 0 0 1 0 18Z", true]],
  mic: [["rect", 9, 3, 6, 11, 3], ["path", "M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"]],
  waveform: [["path", "M4 12h1M8 8.5v7M12 5v14M16 9v6M20 11v2"]],
  minutes: [["path", "M10 6h10M10 12h10M10 18h10M3.5 6l1.2 1.2L7 5M3.5 12l1.2 1.2L7 11M3.5 18l1.2 1.2L7 17"]],
  code: [["path", "m8 7-5 5 5 5M16 7l5 5-5 5"]],
  terminal: [["rect", 3, 4, 18, 16, 3], ["path", "m7.5 9.5 3 2.5-3 2.5M13 15h3.5"]],
  file: [["path", "M6 3h8l4 4v14H6Z"], ["path", "M14 3v4h4"]],
  folder: [["path", "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"]],
  check: [["path", "m5 12.5 4.5 4.5L19 7.5"]],
  copy: [["rect", 9, 9, 11, 11, 2], ["path", "M5 15V6a2 2 0 0 1 2-2h8"]],
  slash: [["path", "M15 4 9 20"]],
  pencil: [["path", "M4 20h4L19.5 8.5a2.1 2.1 0 0 0-3-3L5 17v3Z"]],
  spark: [["path", "M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6 5.6 18.4"]],
  tool: [["path", "M14.5 6.5a4 4 0 0 0-5.3 5.3L4 17v3h3l5.2-5.2a4 4 0 0 0 5.3-5.3l-2.5 2.5-2.5-2.5Z"]],
  compress: [["path", "M4 9h5V4M20 9h-5V4M4 15h5v5M20 15h-5v5"]],
} satisfies Record<string, readonly Shape[]>;

export type IconName = keyof typeof ICONS;

const SVG = "http://www.w3.org/2000/svg";

export function icon(name: IconName, size = 16): SVGSVGElement {
  const svg = document.createElementNS(SVG, "svg");
  for (const [key, value] of Object.entries({
    viewBox: "0 0 24 24",
    width: String(size),
    height: String(size),
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "1.8",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "aria-hidden": "true",
  })) {
    svg.setAttribute(key, value);
  }
  svg.classList.add("icon");

  for (const shape of ICONS[name] as readonly Shape[]) {
    let node: SVGElement;
    let filled: boolean | undefined;
    if (shape[0] === "path") {
      node = document.createElementNS(SVG, "path");
      node.setAttribute("d", shape[1]);
      filled = shape[2];
    } else if (shape[0] === "circle") {
      node = document.createElementNS(SVG, "circle");
      node.setAttribute("cx", String(shape[1]));
      node.setAttribute("cy", String(shape[2]));
      node.setAttribute("r", String(shape[3]));
      filled = shape[4];
    } else {
      node = document.createElementNS(SVG, "rect");
      node.setAttribute("x", String(shape[1]));
      node.setAttribute("y", String(shape[2]));
      node.setAttribute("width", String(shape[3]));
      node.setAttribute("height", String(shape[4]));
      node.setAttribute("rx", String(shape[5]));
      filled = shape[6];
    }
    if (filled) node.setAttribute("fill", "currentColor");
    svg.append(node);
  }
  return svg;
}

/** Fill every `<span data-icon="name">` placeholder in static markup. */
export function hydrateIcons(root: ParentNode): void {
  for (const slot of root.querySelectorAll<HTMLElement>("[data-icon]")) {
    const name = slot.dataset.icon ?? "";
    if (!(name in ICONS)) continue;
    slot.replaceChildren(icon(name as IconName, Number(slot.dataset.size ?? 16)));
    slot.classList.add("icon-slot");
  }
}
