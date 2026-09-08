/**
 * The settings dialog — provider API keys, and the keyboard help.
 *
 * Built on the native `<dialog>` element rather than a positioned div, so the
 * browser supplies the focus trap, Escape-to-close, inertness of the page
 * behind it, and the right semantics for assistive technology. Those are the
 * four things hand-rolled modals reliably get wrong.
 *
 * Three rules the key-entry flow follows, each one a decision rather than a
 * default:
 *
 * **A key is write-only.** The input is always empty on open. There is nothing
 * to read back — the server returns a fingerprint and never the key — so
 * pre-filling would mean either inventing a placeholder that looks like data or
 * asking the server for a secret it should not hand out.
 *
 * **Saving and testing are separate.** Verification costs a request, and a
 * save that silently spends money is a surprise. Test is a button with a
 * visible result.
 *
 * **The failure mode is named.** "Invalid" and "error" are different problems:
 * one means fix your key, the other means check your network. Collapsing them
 * sends people to regenerate a key that was fine.
 */

import { h } from "./dom";
import type { Strings } from "./i18n";
import type { Notifications } from "./toast";

export interface ProviderCredential {
  provider: string;
  label: string;
  modality: string;
  docs_url: string;
  env_var: string;
  configured: boolean;
  source: "environment" | "store" | "none";
  fingerprint: string;
  status: "unknown" | "valid" | "invalid" | "error";
  verified_at: number | null;
  detail: string;
  /** Enumerable form of `detail`; empty when the outcome is a raw error. */
  code: string;
  editable: boolean;
  /** False when this build lacks the vendor SDK, whatever the key says. */
  available: boolean;
}

/**
 * A rejection from the server.
 *
 * `detail` is English and always present; `code` is what lets this client say
 * the same thing in the language the reader chose. Matching on English prose
 * would break the moment the server reworded a sentence.
 */
interface ServerError {
  detail?: string;
  code?: string;
  context?: Record<string, string>;
}

interface CredentialListing {
  providers: ProviderCredential[];
  active_llm: string;
  forced_mock: boolean;
}

export class SettingsDialog {
  private readonly dialog: HTMLDialogElement;
  private readonly body: HTMLElement;
  private listing: CredentialListing | null = null;

  constructor(
    private readonly notify: Notifications,
    private strings: Strings,
    private readonly onChanged: () => void,
  ) {
    this.dialog = h("dialog", { class: "modal", "aria-labelledby": "settings-title" });
    this.body = h("div", { class: "modal-body" });
    this.build();
    document.body.append(this.dialog);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
    const title = this.dialog.querySelector("#settings-title");
    if (title) title.textContent = strings.settings;
    const close = this.dialog.querySelector<HTMLButtonElement>(".modal-foot .btn");
    if (close) close.textContent = strings.close;
    if (this.dialog.open) void this.refresh();
  }

  private build(): void {
    const head = h(
      "div",
      { class: "modal-head" },
      h("h2", { class: "modal-title", id: "settings-title", text: this.strings.settings }),
    );

    const foot = h("div", { class: "modal-foot" });
    const spacer = h("div", { style: "flex:1" });
    const close = h("button", { class: "btn", type: "button", text: this.strings.close });
    close.addEventListener("click", () => this.dialog.close());
    foot.append(spacer, close);

    this.dialog.append(head, this.body, foot);
  }

  async open(): Promise<void> {
    this.dialog.showModal();
    await this.refresh();
  }

  private async refresh(): Promise<void> {
    try {
      const response = await fetch("/v1/credentials");
      if (!response.ok) throw new Error(String(response.status));
      this.listing = (await response.json()) as CredentialListing;
    } catch {
      this.body.replaceChildren(
        h("p", { class: "note", text: "Could not load provider settings." }),
      );
      return;
    }
    this.render();
  }

  private render(): void {
    const s = this.strings;
    const listing = this.listing;
    if (!listing) return;

    const parts: HTMLElement[] = [];

    // What is actually in use right now, stated before any configuration —
    // it is the question someone opening this panel is asking.
    const usingMock = listing.active_llm === "mock" || listing.active_llm.startsWith("mock");
    const active = h("div", { class: "row", style: "margin-bottom:14px" });
    active.append(
      h("span", { class: "note", text: `${s.activeModel}:` }),
      h("span", {
        class: usingMock ? "chip warn" : "chip live",
        text: usingMock ? s.usingMock : listing.active_llm,
      }),
    );
    parts.push(active);
    if (usingMock) {
      parts.push(h("p", { class: "note", style: "margin:-8px 0 14px", text: s.usingMockHint }));
    }

    parts.push(h("h3", { class: "pane-title", style: "margin-bottom:6px", text: s.apiKeys }));
    parts.push(h("p", { class: "note", style: "margin:0 0 14px", text: s.apiKeysHint }));

    for (const provider of listing.providers) {
      parts.push(this.providerCard(provider));
    }

    parts.push(
      h("p", {
        class: "note",
        style: "margin-top:18px;padding-top:14px;border-top:1px solid var(--border)",
        text: s.storageNote,
      }),
    );

    this.body.replaceChildren(...parts);
  }

  private providerCard(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const card = h("div", { class: "provider" });

    const head = h("div", { class: "provider-head" });
    head.append(h("span", { class: "provider-name", text: provider.label }));

    if (provider.configured) {
      head.append(h("span", { class: "provider-key", text: provider.fingerprint }));
      head.append(this.statusChip(provider));
    } else {
      head.append(h("span", { class: "chip", text: s.notConfigured }));
    }
    if (provider.source === "environment") {
      head.append(h("span", { class: "chip", text: s.fromEnvironment }));
    }
    if (!provider.available) {
      head.append(h("span", { class: "chip warn", text: s.sdkMissing }));
    }
    card.append(head);

    // Said before the key field, not after a failed request: a key that is
    // accepted and then quietly ignored is the confusing case.
    if (!provider.available) {
      card.append(h("p", { class: "note", style: "margin:0 0 8px", text: s.sdkMissingHint }));
    }

    const outcome = this.outcome(provider);
    if (outcome && provider.code !== "sdk_missing") {
      card.append(h("p", { class: "note", style: "margin:0 0 8px", text: outcome }));
    }

    if (!provider.editable) {
      card.append(
        h("p", { class: "note", text: `${s.fromEnvironmentHint} (${provider.env_var})` }),
      );
      card.append(this.actions(provider));
      return card;
    }

    const input = h("input", {
      class: "input",
      type: "password",
      placeholder: s.keyPlaceholder,
      autocomplete: "off",
      spellcheck: "false",
      "aria-label": `${provider.label} ${s.apiKeys}`,
    }) as HTMLInputElement;

    const save = h("button", { class: "btn", type: "button", text: s.save }) as HTMLButtonElement;
    save.addEventListener("click", () => void this.save(provider, input, save));
    input.addEventListener("keydown", (event) => {
      if ((event as KeyboardEvent).key === "Enter") void this.save(provider, input, save);
    });

    card.append(h("div", { class: "row", style: "margin-bottom:8px" }, input, save));
    card.append(this.actions(provider));
    return card;
  }

  private statusChip(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const map = {
      valid: { cls: "chip live", text: s.statusValid },
      invalid: { cls: "chip bad", text: s.statusInvalid },
      error: { cls: "chip warn", text: s.statusError },
      unknown: { cls: "chip", text: s.statusUnknown },
    } as const;
    const entry = map[provider.status] ?? map.unknown;
    return h("span", { class: entry.cls, text: entry.text });
  }

  private actions(provider: ProviderCredential): HTMLElement {
    const s = this.strings;
    const row = h("div", { class: "row" });

    const test = h("button", {
      class: "btn ghost",
      type: "button",
      text: s.test,
    }) as HTMLButtonElement;
    test.disabled = !provider.configured;
    test.addEventListener("click", () => void this.verify(provider, test));
    row.append(test);

    if (provider.editable && provider.configured) {
      const remove = h("button", { class: "btn ghost", type: "button", text: s.remove });
      remove.addEventListener("click", () => void this.remove(provider));
      row.append(remove);
    }

    row.append(h("div", { style: "flex:1" }));
    const docs = h("a", {
      class: "note",
      href: provider.docs_url,
      target: "_blank",
      rel: "noopener noreferrer",
      text: s.getKey,
    });
    row.append(docs);
    return row;
  }

  /**
   * A verification outcome in the reader's language.
   *
   * Falls back to the server's English detail when the outcome carried a
   * vendor exception rather than one of the enumerated codes — a specific
   * English message beats a vague translated one.
   */
  private outcome(provider: ProviderCredential): string {
    const s = this.strings;
    const known: Record<string, string> = {
      no_key: s.vNoKey,
      accepted: s.vAccepted,
      rejected: s.vRejected,
      no_permission: s.vNoPermission,
      rate_limited: s.vRateLimited,
      unreachable: s.vUnreachable,
      sdk_missing: s.sdkMissingHint,
    };
    return known[provider.code] ?? provider.detail;
  }

  /** Render a server rejection in the reader's language, or fall back to it. */
  private explain(error: ServerError, status: number): string {
    const s = this.strings;
    const context = error.context ?? {};
    const template =
      error.code === "bad_prefix"
        ? s.errBadPrefix
        : error.code === "empty_key"
          ? s.errEmptyKey
          : error.code === "env_managed"
            ? s.errEnvManaged
            : "";
    if (!template) return error.detail ?? `HTTP ${status}`;
    return template.replace(/\{(\w+)\}/g, (whole, key: string) => context[key] ?? whole);
  }

  // -- actions -------------------------------------------------------------

  private async save(
    provider: ProviderCredential,
    input: HTMLInputElement,
    button: HTMLButtonElement,
  ): Promise<void> {
    const key = input.value.trim();
    if (!key) return;

    button.disabled = true;
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const body = (await response.json().catch(() => ({}))) as ServerError;
      if (!response.ok) {
        input.setAttribute("aria-invalid", "true");
        this.notify.error(this.explain(body, response.status));
        return;
      }
      // Clear immediately: there is no reason for a secret to sit in a DOM
      // node after it has been accepted.
      input.value = "";
      input.removeAttribute("aria-invalid");
      this.notify.ok(`${provider.label} — ${this.strings.saved}`);
      this.onChanged();
      await this.refresh();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      button.disabled = false;
    }
  }

  private async verify(provider: ProviderCredential, button: HTMLButtonElement): Promise<void> {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = this.strings.testing;
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}/verify`, {
        method: "POST",
      });
      const body = await response.json();
      if (!response.ok || !body.provider) {
        this.notify.error(this.explain(body as ServerError, response.status));
        return;
      }
      const updated = body.provider as ProviderCredential;
      const message = `${provider.label} — ${this.outcome(updated)}`;
      if (updated.status === "valid") this.notify.ok(message);
      else if (updated.status === "invalid") this.notify.error(message);
      else this.notify.show(message, "info");
      await this.refresh();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  private async remove(provider: ProviderCredential): Promise<void> {
    try {
      const response = await fetch(`/v1/credentials/${provider.provider}`, { method: "DELETE" });
      if (!response.ok) throw new Error(String(response.status));
      this.notify.ok(`${provider.label} — ${this.strings.removed}`);
      this.onChanged();
      await this.refresh();
    } catch {
      this.notify.error(this.strings.cannotConnect);
    }
  }
}

/** The keyboard-shortcut help sheet. */
export class ShortcutsDialog {
  private readonly dialog: HTMLDialogElement;
  private readonly body: HTMLElement;

  constructor(private strings: Strings) {
    this.dialog = h("dialog", { class: "modal", "aria-labelledby": "shortcuts-title" });
    this.body = h("div", { class: "modal-body" });
    const head = h(
      "div",
      { class: "modal-head" },
      h("h2", { class: "modal-title", id: "shortcuts-title", text: strings.shortcuts }),
    );
    const foot = h("div", { class: "modal-foot" });
    const close = h("button", { class: "btn", type: "button", text: strings.close });
    close.addEventListener("click", () => this.dialog.close());
    foot.append(h("div", { style: "flex:1" }), close);
    this.dialog.append(head, this.body, foot);
    document.body.append(this.dialog);
  }

  setStrings(strings: Strings): void {
    this.strings = strings;
  }

  open(): void {
    const s = this.strings;
    const grid = h("div", { class: "keys" });
    const rows: Array<[string, string]> = [
      ["Space", s.toggleRecord],
      ["/", s.shortcutSearch],
      [",", s.shortcutSettings],
      ["?", s.shortcutHelp],
      ["Esc", s.shortcutClose],
    ];
    for (const [key, description] of rows) {
      grid.append(h("kbd", { text: key }), h("span", { text: description }));
    }
    this.body.replaceChildren(grid);
    this.dialog.showModal();
  }
}
