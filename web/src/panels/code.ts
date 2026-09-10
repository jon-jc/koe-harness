/**
 * The code viewer: a file tree and a syntax-highlighted read-only view.
 *
 * **The highlighter is ~90 lines rather than a library.** Highlight.js is
 * ~120 KB for a client that is currently 20 KB gzipped, to make keywords blue
 * in a panel that exists so you can check what a file says. A tokenizer that
 * handles strings, comments, numbers and keywords covers the languages in this
 * repository and is wrong in ways that are visibly cosmetic rather than
 * misleading. If it ever needs to be right about a language it does not know,
 * that is the moment to take the dependency — not before.
 *
 * **Highlighting builds DOM nodes, never HTML strings.** Source code is
 * arbitrary text, and a file containing `<script>` is a perfectly ordinary
 * file. Every token becomes a text node, so there is no path from file
 * contents to markup at all.
 *
 * **Read-only, and it says so.** Editing needs the read-before-edit policy and
 * an approval path to be safe; shipping the mutation half before the guard is
 * how a tool earns a reputation in one release.
 */

import { h } from "../dom";
import type { Strings } from "../i18n";
import type { Notifications } from "../toast";

interface Entry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
}

interface FileView {
  path: string;
  text: string;
  language: string;
  total_lines: number;
  truncated: boolean;
}

export class CodePanel {
  private readonly tree: HTMLElement;
  private readonly view: HTMLElement;
  private readonly search: HTMLInputElement;
  private readonly expanded = new Set<string>();
  private loaded = false;
  private current = "";

  /**
   * The tree and the file live in different regions now: the tree in the
   * sidebar, where an editor keeps it, and the file in the centre, where
   * there is room to read it.
   */
  constructor(
    hosts: { tree: HTMLElement; view: HTMLElement },
    private strings: Strings,
    private readonly notify: Notifications,
    private readonly events: { onOpen?: (label: string) => void } = {},
  ) {
    this.tree = h("div", { class: "tree", role: "tree" });
    this.view = h("div", { class: "code-view" });
    this.search = h("input", {
      class: "input",
      type: "search",
      placeholder: strings.codeSearch,
      "aria-label": strings.codeSearch,
    }) as HTMLInputElement;

    this.search.addEventListener("keydown", (event) => {
      if (event.key === "Enter") void this.grep(this.search.value.trim());
    });

    hosts.tree.append(h("div", { class: "code-side-head" }, this.search), this.tree);
    hosts.view.append(this.view);
    this.renderEmpty();
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    this.search.placeholder = strings.codeSearch;
    if (!this.current) this.renderEmpty();
  }

  async activate(): Promise<void> {
    if (this.loaded) return;
    this.loaded = true;
    await this.loadDir("");
  }

  private renderEmpty(): void {
    this.view.replaceChildren(
      h("div", { class: "empty" }, h("span", { class: "k", text: "{}" }), this.strings.codeEmpty),
    );
  }

  /* ------------------------------------------------------------------ tree */

  private async loadDir(path: string, into?: HTMLElement): Promise<void> {
    try {
      const response = await fetch(`/v1/fs/tree?path=${encodeURIComponent(path)}`);
      if (!response.ok) throw new Error(String(response.status));
      const body = (await response.json()) as { entries: Entry[] };
      const list = this.renderEntries(body.entries);
      if (into) into.replaceChildren(list);
      else this.tree.replaceChildren(list);
    } catch {
      this.notify.error(this.strings.cannotConnect);
    }
  }

  private renderEntries(entries: Entry[]): HTMLElement {
    const list = h("div", { class: "tree-list", role: "group" });
    for (const entry of entries) {
      const row = h("div", { class: "tree-row" });
      const button = h("button", {
        class: `tree-item${entry.is_dir ? " dir" : ""}`,
        type: "button",
        role: "treeitem",
        "aria-expanded": entry.is_dir ? String(this.expanded.has(entry.path)) : "",
      });
      button.append(
        h("span", { class: "tree-mark", text: entry.is_dir ? "▸" : "" }),
        h("span", { class: "tree-name", text: entry.name }),
      );

      const children = h("div", { class: "tree-children" });
      children.hidden = true;

      button.addEventListener("click", () => {
        if (entry.is_dir) void this.toggle(entry, button, children);
        else void this.openFile(entry.path);
      });

      row.append(button, children);
      list.append(row);
    }
    return list;
  }

  private async toggle(entry: Entry, button: HTMLElement, children: HTMLElement): Promise<void> {
    const open = !children.hidden;
    if (open) {
      children.hidden = true;
      this.expanded.delete(entry.path);
    } else {
      // Loaded on expand rather than up front: walking a repository eagerly
      // costs a request per directory for a tree nobody has looked at.
      if (children.childElementCount === 0) await this.loadDir(entry.path, children);
      children.hidden = false;
      this.expanded.add(entry.path);
    }
    button.setAttribute("aria-expanded", String(!open));
    const mark = button.querySelector(".tree-mark");
    if (mark) mark.textContent = open ? "▸" : "▾";
  }

  /* ------------------------------------------------------------------ file */

  private async openFile(path: string): Promise<void> {
    try {
      const response = await fetch(`/v1/fs/file?path=${encodeURIComponent(path)}`);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        this.notify.error(String(body.detail ?? response.status));
        return;
      }
      this.showFile((await response.json()) as FileView);
    } catch {
      this.notify.error(this.strings.cannotConnect);
    }
  }

  private showFile(file: FileView): void {
    this.current = file.path;
    const head = h("div", { class: "code-head" });
    head.append(
      h("span", { class: "code-path", text: file.path }),
      h("span", { class: "chip", text: file.language }),
      h("span", { class: "note", text: `${file.total_lines.toLocaleString()} ${this.strings.lines}` }),
    );
    if (file.truncated) head.append(h("span", { class: "chip warn", text: this.strings.truncated }));

    const body = h("div", { class: "code-body" });
    const gutter = h("div", { class: "code-gutter", "aria-hidden": "true" });
    const code = h("pre", { class: "code-text" });

    file.text.split("\n").forEach((line, index) => {
      gutter.append(h("span", { text: String(index + 1) }));
      code.append(highlight(line, file.language), document.createTextNode("\n"));
    });

    body.append(gutter, code);
    this.view.replaceChildren(head, body);
    this.events.onOpen?.(file.path);
  }

  /* ---------------------------------------------------------------- search */

  private async grep(query: string): Promise<void> {
    if (!query) return;
    try {
      const response = await fetch(`/v1/fs/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        this.notify.error(String(body.detail ?? response.status));
        return;
      }
      const { hits } = (await response.json()) as {
        hits: Array<{ path: string; line: number; text: string }>;
      };
      const list = h("div", { class: "hits" });
      if (hits.length === 0) {
        list.append(h("p", { class: "note", text: this.strings.noMatches }));
      }
      for (const hit of hits) {
        const row = h("button", { class: "hit", type: "button" });
        row.append(
          h("span", { class: "hit-path", text: `${hit.path}:${hit.line}` }),
          h("span", { class: "hit-text ja", text: hit.text.trim() }),
        );
        row.addEventListener("click", () => void this.openFile(hit.path));
        list.append(row);
      }
      this.view.replaceChildren(
        h("div", { class: "code-head" }, h("span", { class: "code-path", text: `${hits.length} ${this.strings.matches}` })),
        list,
      );
      this.events.onOpen?.(`⌕ ${query}`);
    } catch {
      this.notify.error(this.strings.cannotConnect);
    }
  }
}

/* ---------------------------------------------------------------- highlight */

const KEYWORDS: Record<string, RegExp> = {
  python:
    /\b(def|class|return|if|elif|else|for|while|import|from|as|with|try|except|finally|raise|yield|async|await|lambda|not|and|or|in|is|None|True|False|pass|break|continue|global|nonlocal|assert|del)\b/,
  typescript:
    /\b(const|let|var|function|class|return|if|else|for|while|import|export|from|as|type|interface|extends|implements|new|this|async|await|try|catch|finally|throw|typeof|instanceof|null|undefined|true|false|readonly|private|public|static|enum)\b/,
  json: /\b(true|false|null)\b/,
};
KEYWORDS.tsx = KEYWORDS.typescript;
KEYWORDS.javascript = KEYWORDS.typescript;
KEYWORDS.jsx = KEYWORDS.typescript;

const COMMENTS: Record<string, RegExp> = {
  python: /#/,
  typescript: /\/\//,
  tsx: /\/\//,
  javascript: /\/\//,
  jsx: /\/\//,
  css: /\/\//,
  rust: /\/\//,
  go: /\/\//,
  yaml: /#/,
  toml: /#/,
  bash: /#/,
  ini: /[#;]/,
};

/**
 * One line, as a document fragment of spans.
 *
 * A single left-to-right pass rather than a set of regexes applied in turn,
 * because overlapping matches are where naive highlighters go wrong: a keyword
 * inside a string, or a `#` inside a URL inside a comment. Consuming the line
 * once means each character belongs to exactly one token by construction.
 */
function highlight(line: string, language: string): DocumentFragment {
  const fragment = document.createDocumentFragment();
  const keywords = KEYWORDS[language];
  const comment = COMMENTS[language];

  let index = 0;
  let plain = "";

  const flush = (): void => {
    if (plain) {
      fragment.append(document.createTextNode(plain));
      plain = "";
    }
  };
  const token = (text: string, kind: string): void => {
    flush();
    fragment.append(h("span", { class: `t-${kind}`, text }));
  };

  while (index < line.length) {
    const rest = line.slice(index);

    if (comment && comment.test(rest[0] + (rest[1] ?? "")) && comment.exec(rest)?.index === 0) {
      token(rest, "comment");
      return fragment;
    }

    const quote = rest[0];
    if (quote === '"' || quote === "'" || quote === "`") {
      // Scan to the matching quote, honouring backslash escapes, so a string
      // containing a quote does not end the token early.
      let end = 1;
      while (end < rest.length) {
        if (rest[end] === "\\") end += 2;
        else if (rest[end] === quote) break;
        else end += 1;
      }
      const text = rest.slice(0, Math.min(end + 1, rest.length));
      token(text, "string");
      index += text.length;
      continue;
    }

    const word = /^[A-Za-z_$][A-Za-z0-9_$]*/.exec(rest);
    if (word) {
      const text = word[0];
      if (keywords && new RegExp(`^${keywords.source}$`).test(text)) token(text, "keyword");
      else plain += text;
      index += text.length;
      continue;
    }

    const number = /^\d[\d_.xXa-fA-F]*/.exec(rest);
    if (number) {
      token(number[0], "number");
      index += number[0].length;
      continue;
    }

    plain += rest[0];
    index += 1;
  }

  flush();
  return fragment;
}
