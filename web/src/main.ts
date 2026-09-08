/**
 * The live transcription UI.
 *
 * One rendering decision does most of the work here: **committed and pending
 * text are styled differently**. The server tells us which part of an interim
 * hypothesis has settled (agreed across successive decodes) and which part is
 * still provisional. Rendering both identically produces a line that visibly
 * rewrites itself several times a second, which is genuinely hard to read —
 * the eye keeps re-reading a line that keeps changing. Greying the unsettled
 * tail lets a reader follow the settled text while the tail churns.
 */

import { MicrophoneCapture } from './audio';
import { StreamClient, type ServerMessage } from './stream-client';

type Language = 'ja' | 'en' | 'unknown';

interface FinalSegment {
  text: string;
  start: number;
  end: number;
  speaker: string;
}

const STRINGS = {
  ja: {
    idle: '待機中',
    connecting: '接続中…',
    listening: '認識中',
    stopped: '停止しました',
    start: '録音開始',
    stop: '停止',
    empty: 'マイクを開始すると、ここに文字起こしが表示されます。',
    speaking: '発話中',
    silence: '無音',
  },
  en: {
    idle: 'idle',
    connecting: 'connecting…',
    listening: 'listening',
    stopped: 'stopped',
    start: 'Start',
    stop: 'Stop',
    empty: 'Start the microphone and the transcript will appear here.',
    speaking: 'speech',
    silence: 'silence',
  },
} as const;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

class App {
  private client: StreamClient | null = null;
  private capture: MicrophoneCapture | null = null;
  private finals: FinalSegment[] = [];
  private committed = '';
  private pending = '';
  private language: Language = 'ja';
  private running = false;
  private startedAt = 0;

  private readonly toggle = el<HTMLButtonElement>('toggle');
  private readonly status = el<HTMLSpanElement>('status');
  private readonly provider = el<HTMLSpanElement>('provider');
  private readonly meter = el<HTMLDivElement>('meter-fill');
  private readonly transcript = el<HTMLDivElement>('transcript');
  private readonly stats = el<HTMLDivElement>('stats');
  private readonly languageSelect = el<HTMLSelectElement>('language');
  private readonly speech = el<HTMLSpanElement>('speech-state');

  constructor() {
    this.toggle.addEventListener('click', () => void this.onToggle());
    this.languageSelect.addEventListener('change', () => {
      this.language = this.languageSelect.value as Language;
      this.render();
    });
    this.render();
  }

  private get strings() {
    return this.language === 'en' ? STRINGS.en : STRINGS.ja;
  }

  private async onToggle(): Promise<void> {
    if (this.running) {
      await this.stop();
    } else {
      await this.start();
    }
  }

  private async start(): Promise<void> {
    this.finals = [];
    this.committed = '';
    this.pending = '';
    this.setStatus(this.strings.connecting);

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    this.client = new StreamClient(`${protocol}://${location.host}/v1/stream`, {
      onMessage: (message) => this.onMessage(message),
      onError: (error) => this.setStatus(`error: ${error}`),
      onClose: () => {
        if (this.running) void this.stop();
      },
    });

    try {
      await this.client.connect();
    } catch {
      this.setStatus('could not connect');
      return;
    }

    this.client.start({ language: this.language, partialIntervalMs: 400 });

    this.capture = new MicrophoneCapture({
      onFrame: (frame) => this.client?.sendAudio(frame),
      onLevel: (level) => {
        this.meter.style.width = `${Math.min(100, level * 140)}%`;
      },
    });

    try {
      await this.capture.start();
    } catch {
      // The overwhelmingly common cause is a denied permission prompt, and the
      // browser gives no way to distinguish that from a missing device.
      this.setStatus('microphone unavailable — check permissions');
      this.client.close();
      return;
    }

    this.running = true;
    this.startedAt = performance.now();
    this.toggle.textContent = this.strings.stop;
    this.toggle.classList.add('recording');
    this.setStatus(this.strings.listening);
    this.render();
  }

  private async stop(): Promise<void> {
    this.running = false;
    this.toggle.textContent = this.strings.start;
    this.toggle.classList.remove('recording');
    this.setStatus(this.strings.stopped);
    this.meter.style.width = '0%';

    await this.capture?.stop();
    this.capture = null;

    this.client?.stop();
    // Give the server a moment to send the closing transcript before the
    // socket goes away, otherwise the final segment is lost on every session.
    setTimeout(() => this.client?.close(), 1500);
  }

  private onMessage(message: ServerMessage): void {
    switch (message.type) {
      case 'started':
        this.provider.textContent = message.provider;
        break;
      case 'speech':
        this.speech.textContent =
          message.state === 'start' ? this.strings.speaking : this.strings.silence;
        this.speech.className = message.state === 'start' ? 'badge active' : 'badge';
        break;
      case 'partial':
        this.committed = message.committed;
        this.pending = message.pending;
        this.render();
        break;
      case 'final':
        this.finals.push({
          text: message.text,
          start: message.start,
          end: message.end,
          speaker: message.speaker,
        });
        this.committed = '';
        this.pending = '';
        this.render();
        break;
      case 'transcript':
        this.stats.textContent = `${message.duration.toFixed(1)}s · $${message.cost_usd.toFixed(5)} · ${this.client?.sentFrames ?? 0} frames sent, ${this.client?.droppedFrames ?? 0} dropped`;
        break;
      case 'error':
        this.setStatus(`error: ${message.message}`);
        break;
    }
  }

  private setStatus(text: string): void {
    this.status.textContent = text;
  }

  private render(): void {
    this.toggle.textContent = this.running ? this.strings.stop : this.strings.start;

    if (this.finals.length === 0 && !this.committed && !this.pending) {
      this.transcript.innerHTML = `<p class="empty">${this.strings.empty}</p>`;
      return;
    }

    const parts: string[] = [];
    for (const segment of this.finals) {
      const speaker = segment.speaker
        ? `<span class="speaker">${escapeHtml(segment.speaker)}</span>`
        : '';
      parts.push(
        `<p class="line final">${speaker}<span class="text">${escapeHtml(segment.text)}</span>` +
          `<span class="time">${segment.start.toFixed(1)}s</span></p>`,
      );
    }

    if (this.committed || this.pending) {
      // Settled text renders normally; the tail the model is still deciding on
      // is greyed, so the reader can tell what is safe to read.
      parts.push(
        `<p class="line interim"><span class="text">${escapeHtml(this.committed)}` +
          `<span class="pending">${escapeHtml(this.pending)}</span></span></p>`,
      );
    }

    this.transcript.innerHTML = parts.join('');
    this.transcript.scrollTop = this.transcript.scrollHeight;

    if (this.running) {
      const elapsed = (performance.now() - this.startedAt) / 1000;
      this.stats.textContent = `${elapsed.toFixed(1)}s · ${this.finals.length} segments`;
    }
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

document.addEventListener('DOMContentLoaded', () => {
  new App();
});
