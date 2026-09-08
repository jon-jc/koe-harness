/**
 * Transient notifications.
 *
 * A stack rather than a single slot, because two things can fail at once and
 * the second replacing the first means the user never sees it — which is
 * exactly when they most needed to.
 *
 * Errors do not auto-dismiss. A success message vanishing is fine; a failure
 * vanishing before it is read leaves someone with a system that did not do
 * what they asked and no explanation.
 */

import { h } from "./dom";

export type ToastKind = "error" | "ok" | "info";

const AUTO_DISMISS_MS: Record<ToastKind, number> = {
  ok: 3200,
  info: 4500,
  error: 0, // stays until dismissed
};

export class Notifications {
  private readonly host: HTMLElement;
  private readonly live: HTMLElement;

  constructor(host?: HTMLElement, live?: HTMLElement) {
    this.host = host ?? this.createHost();
    this.live =
      live ??
      (document.getElementById("live-region") as HTMLElement | null) ??
      this.createHost();
  }

  private createHost(): HTMLElement {
    const existing = document.querySelector<HTMLElement>("body > .toasts");
    if (existing) return existing;
    const node = h("div", { class: "toasts" });
    document.body.append(node);
    return node;
  }

  /**
   * Where this notification should go.
   *
   * A modal `<dialog>` renders in the **top layer**, which sits above every
   * z-index on the page — so a toast in the body-level host would be painted
   * behind the very form it is reporting on. Position is inherited from a
   * descendant of the dialog, so hosting it there puts it back in front.
   * Closing the dialog takes its notifications with it, which is right: they
   * were about the dialog.
   */
  private hostFor(): HTMLElement {
    const dialog = document.querySelector<HTMLElement>("dialog[open]");
    if (!dialog) return this.host;
    const existing = dialog.querySelector<HTMLElement>(":scope > .toasts");
    if (existing) return existing;
    const node = h("div", { class: "toasts" });
    dialog.append(node);
    return node;
  }

  show(message: string, kind: ToastKind = "info", dismissLabel = "Dismiss"): void {
    const toast = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" });
    toast.append(h("span", { text: message }));

    const close = h("button", {
      class: "close",
      type: "button",
      "aria-label": dismissLabel,
      text: "×",
    });
    close.addEventListener("click", () => toast.remove());
    toast.append(close);

    this.hostFor().append(toast);

    // Errors are announced assertively; routine confirmations should not
    // interrupt whatever a screen reader is currently saying.
    this.live.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");
    this.live.textContent = message;

    const timeout = AUTO_DISMISS_MS[kind];
    if (timeout > 0) window.setTimeout(() => toast.remove(), timeout);
  }

  error(message: string): void {
    this.show(message, "error");
  }

  ok(message: string): void {
    this.show(message, "ok");
  }

  info(message: string): void {
    this.show(message, "info");
  }

  clear(): void {
    for (const host of document.querySelectorAll(".toasts")) host.replaceChildren();
  }
}
