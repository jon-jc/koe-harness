/**
 * WebSocket client for the koe realtime protocol.
 *
 * Control frames are JSON, audio frames are binary. The split matters on the
 * audio path: base64-ing PCM into JSON inflates every frame by a third, on the
 * one channel where traffic is continuous rather than occasional.
 *
 * The client drops audio when the socket's send buffer backs up rather than
 * queueing it. Queueing feels safer and is worse: the queue grows without
 * bound on a slow link, and the audio that eventually drains is minutes stale,
 * so the user watches captions for speech they have long since finished.
 * Dropping keeps the stream anchored to now.
 */

export type ServerMessage =
  | { type: 'started'; session_id: string; provider: string; language: string }
  | { type: 'speech'; state: 'start' | 'end'; at: number }
  | { type: 'partial'; committed: string; pending: string; language: string }
  | { type: 'final'; text: string; start: number; end: number; language: string; speaker: string }
  | {
      type: 'transcript';
      text: string;
      duration: number;
      cost_usd: number;
      segments: Array<{ text: string; start: number; end: number; speaker: string }>;
    }
  | {
      type: 'minutes';
      minutes: MinutesPayload;
      rendered: string;
      grounded: number;
      total_claims: number;
      dropped: string[];
      repairs: number;
      cost_usd: number;
    }
  | { type: 'level'; value: number }
  | { type: 'demo_finished' }
  | { type: 'error'; message: string };

/** The subset of the server's minutes model the client renders. */
export interface MinutesPayload {
  title: string;
  participants: string[];
  summary: string;
  topics: Array<{ title: string; summary: string }>;
  decisions: Array<{ statement: string; source_quote: string; speaker: string }>;
  action_items: Array<{
    task: string;
    owner: string;
    due: string;
    source_quote: string;
    speaker: string;
  }>;
}

export interface StreamHandlers {
  onMessage: (message: ServerMessage) => void;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: string) => void;
}

export interface StartOptions {
  language: 'ja' | 'en' | 'unknown';
  partialIntervalMs?: number;
  /**
   * Endpointing overrides.
   *
   * Both are omitted unless the user has turned off the language defaults.
   * Sending a number here replaces a per-language default that matters: the
   * Japanese silence window is longer than the English one because speakers
   * pause before sentence-final particles, and an English-tuned endpointer
   * cuts there — removing the verb, which is where Japanese carries negation
   * and tense. The server clamps whatever arrives.
   */
  silenceToEndMs?: number;
  speechThresholdDb?: number;
}

/** Above this many buffered bytes, new frames are dropped rather than queued. */
const BACKPRESSURE_BYTES = 512 * 1024;

export class StreamClient {
  private socket: WebSocket | null = null;
  private dropped = 0;
  private sent = 0;

  constructor(
    private readonly url: string,
    private readonly handlers: StreamHandlers,
  ) {}

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /** Frames discarded to back-pressure — surfaced so degradation is visible. */
  get droppedFrames(): number {
    return this.dropped;
  }

  get sentFrames(): number {
    return this.sent;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      socket.binaryType = 'arraybuffer';
      this.socket = socket;

      socket.onopen = () => {
        this.handlers.onOpen?.();
        resolve();
      };

      socket.onmessage = (event: MessageEvent) => {
        if (typeof event.data !== 'string') return;
        try {
          this.handlers.onMessage(JSON.parse(event.data) as ServerMessage);
        } catch {
          this.handlers.onError?.('received a malformed frame from the server');
        }
      };

      socket.onerror = () => {
        this.handlers.onError?.('connection failed');
        reject(new Error('websocket error'));
      };

      socket.onclose = (event: CloseEvent) => {
        this.socket = null;
        this.handlers.onClose?.(event);
      };
    });
  }

  start(options: StartOptions): void {
    this.send({
      type: 'start',
      language: options.language,
      partial_interval_ms: options.partialIntervalMs ?? 500,
      ...(options.silenceToEndMs === undefined
        ? {}
        : { silence_to_end_ms: options.silenceToEndMs }),
      ...(options.speechThresholdDb === undefined
        ? {}
        : { speech_threshold_db: options.speechThresholdDb }),
    });
  }

  /** Send one PCM16 frame, dropping it if the socket is already backed up. */
  sendAudio(frame: ArrayBuffer): void {
    if (!this.connected || !this.socket) return;
    if (this.socket.bufferedAmount > BACKPRESSURE_BYTES) {
      this.dropped += 1;
      return;
    }
    this.socket.send(frame);
    this.sent += 1;
  }

  stop(): void {
    this.send({ type: 'stop' });
  }

  /** Ask for minutes over the same socket the transcript arrived on. */
  requestMinutes(): void {
    this.send({ type: 'minutes' });
  }

  /** Ask the server to play a scripted meeting through the real pipeline. */
  startDemo(meeting: string, language: StartOptions['language']): void {
    this.send({ type: 'demo', meeting, language });
  }

  close(): void {
    // Tell the server first so it can tear the session scope down cleanly,
    // rather than discovering the disconnect from a failed write.
    this.send({ type: 'close' });
    this.socket?.close();
    this.socket = null;
  }

  private send(payload: Record<string, unknown>): void {
    if (!this.connected || !this.socket) return;
    this.socket.send(JSON.stringify(payload));
  }
}
