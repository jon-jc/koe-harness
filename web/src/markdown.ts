/**
 * Just enough Markdown for an agent's answers.
 *
 * Models answer in Markdown whether asked to or not, and showing the asterisks
 * makes a correct answer look broken. A library would be 30-40 KB for a client
 * this size, and the part of Markdown answers actually use is small: fenced
 * code, inline code, bold, lists, headings and paragraphs.
 *
 * **It builds DOM nodes, never an HTML string.** A model's output is untrusted
 * text — it routinely quotes files, and a file can contain `<img onerror>` —
 * so every piece of it becomes a text node and there is no path from the
 * answer to markup at all.
 *
 * Anything it does not recognise is left as the literal text it was, which is
 * the right failure: slightly less decorated, never wrong.
 */

import { h } from "./dom";

const FENCE = /^```\s*([\w+-]*)\s*$/;
const BULLET = /^\s*[-*]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const HEADING = /^(#{1,4})\s+(.*)$/;

export function renderMarkdown(text: string): DocumentFragment {
  const out = document.createDocumentFragment();
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let paragraph: string[] = [];
  let list: HTMLElement | null = null;

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    const node = h("p", { class: "md-p ja" });
    paragraph.forEach((line, index) => {
      if (index > 0) node.append(h("br"));
      inline(node, line);
    });
    out.append(node);
    paragraph = [];
  };
  const flushList = () => {
    if (list) out.append(list);
    list = null;
  };

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];

    const fence = FENCE.exec(line);
    if (fence) {
      flushParagraph();
      flushList();
      const body: string[] = [];
      index += 1;
      // An unterminated fence runs to the end, which is what a model that was
      // cut off mid-answer produced, and still reads as code.
      while (index < lines.length && !FENCE.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      const block = h("div", { class: "md-code" });
      if (fence[1]) block.append(h("span", { class: "md-lang", text: fence[1] }));
      block.append(h("pre", { text: body.join("\n") }));
      out.append(block);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const node = h("p", { class: `md-h md-h${heading[1].length} ja` });
      inline(node, heading[2]);
      out.append(node);
      continue;
    }

    const bullet = BULLET.exec(line);
    const numbered = bullet ? null : NUMBERED.exec(line);
    if (bullet || numbered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      const current = list as HTMLElement | null;
      if (!current || current.tagName.toLowerCase() !== tag) {
        flushList();
        list = h(tag, { class: "md-list ja" });
      }
      const item = h("li");
      inline(item, (bullet ?? numbered)![1]);
      (list as HTMLElement).append(item);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  return out;
}

/** `code` and **bold**, in one left-to-right pass. */
function inline(host: HTMLElement, text: string): void {
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let at = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    if (start > at) host.append(text.slice(at, start));
    const token = match[0];
    if (token.startsWith("`")) host.append(h("code", { class: "md-inline", text: token.slice(1, -1) }));
    else host.append(h("strong", { text: token.slice(2, -2) }));
    at = start + token.length;
  }
  if (at < text.length) host.append(text.slice(at));
}
