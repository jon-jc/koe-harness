/**
 * Microphone capture and conversion to the wire format.
 *
 * The server wants 16 kHz mono PCM16, which is what ASR models want. Browsers
 * give you 48 kHz float32. Converting in the browser rather than on the server
 * is a deliberate cost decision: it moves the resampling onto the client's CPU
 * (which is idle) and cuts uplink bandwidth ~6x, which matters because this is
 * the one path where bandwidth is continuous rather than occasional.
 *
 * The conversion runs inside an **AudioWorklet**, not a ScriptProcessorNode.
 * ScriptProcessorNode runs on the main thread, so a React re-render or a
 * garbage collection pause drops audio frames — and dropped input audio is
 * unrecoverable in a way that a dropped UI frame is not. The worklet runs on
 * the audio thread, which is real-time scheduled.
 */

/** How the worklet is configured from the main thread. */
export interface CaptureOptions {
  /** Target sample rate for the wire format. */
  targetSampleRate?: number;
  /** Frame size in milliseconds sent per message. */
  frameMs?: number;
  /** Called with each PCM16 frame, ready to put on the socket. */
  onFrame: (frame: ArrayBuffer) => void;
  /** Called with a 0..1 level, for the UI meter. */
  onLevel?: (level: number) => void;
}

/**
 * The worklet source, inlined so there is no second file to serve and no way
 * for the two to drift out of sync.
 */
const WORKLET_SOURCE = /* js */ `
class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetSampleRate || 16000;
    this.frameSamples = Math.round(this.targetRate * (opts.frameMs || 20) / 1000);
    this.ratio = sampleRate / this.targetRate;
    this.buffer = new Int16Array(this.frameSamples);
    this.filled = 0;
    this.cursor = 0;
    this.peak = 0;
    this.sinceLevel = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    // Decimate by averaging over each source window. Averaging is a crude
    // low-pass, but dropping samples outright aliases high-frequency content
    // down into the speech band, which sounds like a metallic buzz and
    // measurably degrades recognition.
    while (this.cursor < channel.length) {
      const start = this.cursor;
      const end = Math.min(channel.length, start + this.ratio);
      let sum = 0;
      let count = 0;
      for (let i = Math.floor(start); i < Math.ceil(end) && i < channel.length; i++) {
        sum += channel[i];
        count++;
      }
      const value = count > 0 ? sum / count : 0;
      const magnitude = Math.abs(value);
      if (magnitude > this.peak) this.peak = magnitude;

      const clamped = Math.max(-1, Math.min(1, value));
      this.buffer[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this.filled >= this.frameSamples) {
        // Copy: the buffer is reused immediately, and a transferred view of it
        // would be detached out from under the next frame.
        const frame = this.buffer.slice(0);
        this.port.postMessage({ type: 'frame', frame: frame.buffer }, [frame.buffer]);
        this.filled = 0;
      }
      this.cursor = end;
    }
    this.cursor -= channel.length;

    this.sinceLevel += channel.length;
    if (this.sinceLevel >= sampleRate / 20) {
      this.port.postMessage({ type: 'level', level: this.peak });
      this.peak = 0;
      this.sinceLevel = 0;
    }
    return true;
  }
}
registerProcessor('pcm-downsampler', PCMDownsampler);
`;

/** Captures the microphone and emits PCM16 frames at the target rate. */
export class MicrophoneCapture {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private moduleUrl: string | null = null;

  constructor(private readonly options: CaptureOptions) {}

  get active(): boolean {
    return this.context !== null;
  }

  async start(): Promise<void> {
    if (this.context) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // The browser's own processing is better than anything we would add,
        // and echo cancellation in particular is required for laptop speakers.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.context = new AudioContext();
    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    this.moduleUrl = URL.createObjectURL(blob);
    await this.context.audioWorklet.addModule(this.moduleUrl);

    this.node = new AudioWorkletNode(this.context, 'pcm-downsampler', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: {
        targetSampleRate: this.options.targetSampleRate ?? 16000,
        frameMs: this.options.frameMs ?? 20,
      },
    });

    this.node.port.onmessage = (event: MessageEvent) => {
      const data = event.data as { type: string; frame?: ArrayBuffer; level?: number };
      if (data.type === 'frame' && data.frame) {
        this.options.onFrame(data.frame);
      } else if (data.type === 'level' && this.options.onLevel) {
        this.options.onLevel(data.level ?? 0);
      }
    };

    this.source = this.context.createMediaStreamSource(this.stream);
    this.source.connect(this.node);
  }

  async stop(): Promise<void> {
    // Release in reverse order of acquisition. In particular the MediaStream
    // tracks must be stopped explicitly, or the browser keeps showing the
    // recording indicator after the session has ended.
    this.node?.port.close();
    this.source?.disconnect();
    this.node?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context) await this.context.close();
    if (this.moduleUrl) URL.revokeObjectURL(this.moduleUrl);

    this.node = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.moduleUrl = null;
  }
}
