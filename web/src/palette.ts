/**
 * The command palette.
 *
 * Four workspaces, a settings panel with six sections, two terminals and a
 * dozen tools is more surface than a menu bar can hold and more than anyone
 * will memorise shortcuts for. A palette is the standard answer, and it earns
 * its place here for a reason specific to this app: **it is the only control
 * that is equally reachable in both languages.** Someone who reads Japanese
 * types 議事録 and someone who reads English types "minutes"; both match the
 * same command, because every entry carries both labels and the filter checks
 * each.
 *
 * **Matching is subsequence, not substring.** "gm" finds "Generate minutes"
 * and "tt" finds "Toggle theme" — the shorthand people actually type once they
 * know a palette exists. A substring filter makes them type the beginning of a
 * word they may not remember the exact wording of.
 *
 * **Ranking prefers a prefix, then a word boundary, then anything.** Without
 * that, a short query surfaces whichever command happens to sort first, and
 * the palette feels arbitrary rather than predictive.
 */

import { h } from "./dom";
import type { Strings, UILang } from "./i18n";

export interface Command {
  id: string;
  /** Both labels, always: the palette is the bilingual entry point. */
  ja: string;
  en: string;
  /** Where it goes, shown dimmed after the label. */
  group?: string;
  /** Rendered as a key hint on the right. */
  hint?: string;
  run: () => void;
}

interface Scored {
  command: Command;
  score: number;
  label: string;
}

export class CommandPalette {
  private readonly dialog: HTMLDialogElement;
  private readonly input: HTMLInputElement;
  private readonly list: HTMLElement;
  private commands: Command[] = [];
  private matches: Scored[] = [];
  private active = 0;

  constructor(
    private strings: Strings,
    private lang: () => UILang,
  ) {
    this.dialog = h("dialog", { class: "modal palette", "aria-label": "commands" });
    this.input = h("input", {
      class: "palette-input",
      type: "text",
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": strings.paletteSearch,
      placeholder: strings.paletteSearch,
    }) as HTMLInputElement;
    this.list = h("div", { class: "palette-list", role: "listbox" });

    this.dialog.append(
      h("div", { class: "palette-head" }, this.input),
      this.list,
      h("div", { class: "palette-foot note" }, h("span", { text: strings.paletteHint })),
    );
    document.body.append(this.dialog);
    this.bind();
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.input.placeholder = strings.paletteSearch;
    this.input.setAttribute("aria-label", strings.paletteSearch);
    const foot = this.dialog.querySelector(".palette-foot span");
    if (foot) foot.textContent = strings.paletteHint;
  }

  register(commands: Command[]): void {
    this.commands = commands;
  }

  get open(): boolean {
    return this.dialog.open;
  }

  show(): void {
    this.input.value = "";
    this.filter("");
    this.dialog.showModal();
    this.input.focus();
  }

  private bind(): void {
    this.input.addEventListener("input", () => this.filter(this.input.value));

    this.input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || (event.key === "n" && event.ctrlKey)) {
        event.preventDefault();
        this.move(1);
      } else if (event.key === "ArrowUp" || (event.key === "p" && event.ctrlKey)) {
        event.preventDefault();
        this.move(-1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        this.invoke(this.active);
      }
    });

    // Escape closes it explicitly, not only natively: a <dialog>'s built-in
    // Escape handling is skipped by some embedded webviews and synthesized
    // input, and a modal that cannot be dismissed from the keyboard traps
    // whoever opened it.
    this.dialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault();
        this.dialog.close();
      }
    });

    // Clicking the backdrop closes it. A palette is a transient thing and
    // trapping someone in it because they clicked past it is hostile.
    this.dialog.addEventListener("click", (event) => {
      if (event.target === this.dialog) this.dialog.close();
    });
  }

  private filter(query: string): void {
    const trimmed = query.trim();
    const lang = this.lang();

    this.matches = this.commands
      .map((command) => {
        const label = lang === "ja" ? command.ja : command.en;
        // Both labels are searched whichever way the UI is set, so a
        // Japanese speaker on an English interface still finds things by
        // typing Japanese — and vice versa.
        const score = Math.max(
          rank(command.ja, trimmed),
          rank(command.en, trimmed),
          rank(command.id, trimmed),
        );
        return { command, score, label };
      })
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score);

    this.active = 0;
    this.render();
  }

  private render(): void {
    const s = this.strings;
    if (this.matches.length === 0) {
      this.list.replaceChildren(h("div", { class: "palette-empty note", text: s.noMatches }));
      return;
    }

    this.list.replaceChildren(
      ...this.matches.map((entry, index) => {
        const row = h("button", {
          class: `palette-item${index === this.active ? " on" : ""}`,
          type: "button",
          role: "option",
          "aria-selected": String(index === this.active),
        });
        row.append(h("span", { class: "palette-label", text: entry.label }));
        if (entry.command.group) {
          row.append(h("span", { class: "palette-group", text: entry.command.group }));
        }
        row.append(h("span", { style: "flex:1" }));
        if (entry.command.hint) {
          row.append(h("kbd", { text: entry.command.hint }));
        }
        row.addEventListener("click", () => this.invoke(index));
        // Hovering moves the selection, so mouse and keyboard do not fight
        // over which row Enter would run.
        row.addEventListener("mousemove", () => {
          if (this.active !== index) {
            this.active = index;
            this.render();
          }
        });
        return row;
      }),
    );
  }

  private move(step: number): void {
    if (this.matches.length === 0) return;
    this.active = (this.active + step + this.matches.length) % this.matches.length;
    this.render();
    this.list.children[this.active]?.scrollIntoView({ block: "nearest" });
  }

  private invoke(index: number): void {
    const entry = this.matches[index];
    if (!entry) return;
    // Closed first: a command that opens another dialog cannot do so while
    // this one still owns the top layer.
    this.dialog.close();
    entry.command.run();
  }
}

/**
 * How well `text` matches `query`, as a score. 0 means no match.
 *
 * An empty query matches everything at a low score, so opening the palette
 * shows the full list rather than nothing.
 */
function rank(text: string, query: string): number {
  if (!query) return 1;
  const haystack = text.toLowerCase();
  const needle = query.toLowerCase();

  if (haystack.startsWith(needle)) return 1000;
  const at = haystack.indexOf(needle);
  if (at === 0) return 1000;
  if (at > 0) {
    // A match at a word boundary is what someone meant more often than one
    // buried mid-word.
    return haystack[at - 1] === " " ? 800 : 600;
  }
  return subsequence(haystack, needle);
}

/** Score a subsequence match: every needle character in order, gaps allowed. */
function subsequence(haystack: string, needle: string): number {
  let index = 0;
  let score = 0;
  let previous = -1;

  for (const character of needle) {
    const found = haystack.indexOf(character, index);
    if (found === -1) return 0;
    // Adjacent characters score higher than scattered ones, so "gm" prefers
    // "Generate minutes" over a command that merely contains a g and an m.
    score += found === previous + 1 ? 10 : 2;
    previous = found;
    index = found + 1;
  }
  return score;
}
