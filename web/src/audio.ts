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
 *
 * ## Three sources, because meetings are not all in one room
 *
 * A minutes tool that can only hear a microphone can only take minutes of an
 * in-person meeting. Most meetings are a call, where the people who matter are
 * coming out of the speakers. So capture has three modes:
 *
 *   - **microphone** — a chosen input device.
 *   - **system** — `getDisplayMedia`, whose picker lets the user choose what to
 *     listen to. The video track is stopped the moment it arrives; only the
 *     audio is wanted, and an unused video track is pure CPU and battery.
 *   - **both** — mixed through a common `GainNode`, which is what a hybrid
 *     meeting needs: your voice from the microphone, the remote room from the
 *     system.
 *
 * The picker's audio checkbox is offered for a screen or a browser tab, and on
 * Windows *not* for a single application window — a Chromium limitation, not a
 * choice made here. `systemAudioHint()` reports which one the user actually
 * granted so the UI can say "that share has no audio" instead of transcribing
 * silence.
 */

/** Where the audio comes from. */
export type AudioSource = "microphone" | "system" | "both";

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
  /** Which input to open. Defaults to the microphone. */
  source?: AudioSource;
  /** `deviceId` from `listInputDevices`; empty means the system default. */
  deviceId?: string;
  /** Linear gain applied after capture. */
  gain?: number;
  /** Browser-side processing. Wrong for a room mic or for music. */
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  /** Called when a shared source ends — the user pressed "Stop sharing". */
  onSourceEnded?: () => void;
}

export interface InputDevice {
  readonly deviceId: string;
  readonly label: string;
}

/**
 * The microphones this browser will admit to having.
 *
 * Labels are empty until the page holds a media permission — the browser
 * withholds them because the device list is otherwise a fingerprinting vector.
 * Callers should enumerate again after a successful capture, when the names
 * appear.
 */
export async function listInputDevices(): Promise<InputDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((device) => device.kind === "audioinput")
      .map((device, index) => ({
        deviceId: device.deviceId,
        label: device.label || `Microphone ${index + 1}`,
      }));
  } catch {
    return [];
  }
}

/** Whether this browser can capture system audio at all. */
export function canCaptureSystemAudio(): boolean {
  return typeof navigator.mediaDevices?.getDisplayMedia === "function";
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

/** Raised when a capture cannot start, with a reason the UI can name. */
export class CaptureError extends Error {
  constructor(
    readonly reason: "denied" | "no-audio" | "unsupported" | "unavailable",
    message: string,
  ) {
    super(message);
    this.name = "CaptureError";
  }
}

/** Captures audio from one or two sources and emits PCM16 at the target rate. */
export class AudioCapture {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private gain: GainNode | null = null;
  private readonly inputs: MediaStreamAudioSourceNode[] = [];
  private readonly streams: MediaStream[] = [];
  private moduleUrl: string | null = null;
  /** What the user actually granted, for a UI that should not have to guess. */
  private granted = { microphone: false, system: false };

  constructor(private readonly options: CaptureOptions) {}

  get active(): boolean {
    return this.context !== null;
  }

  /** Which sources are live. `system` is false after a share with no audio. */
  get sources(): { microphone: boolean; system: boolean } {
    return { ...this.granted };
  }

  /** Change gain mid-session; a volume slider should not require a restart. */
  setGain(value: number): void {
    if (this.gain) this.gain.gain.value = value;
  }

  async start(): Promise<void> {
    if (this.context) return;
    const source = this.options.source ?? "microphone";

    // Acquire before building the graph, so a refused permission leaves
    // nothing behind to tear down.
    const captured: MediaStream[] = [];
    try {
      if (source === "microphone" || source === "both") {
        captured.push(await this.openMicrophone());
        this.granted.microphone = true;
      }
      if (source === "system" || source === "both") {
        const system = await this.openSystemAudio();
        // A single-window share on Windows carries no audio. Mixed with a
        // microphone that is a degraded session worth continuing; on its own
        // it is silence, and transcribing silence while appearing to work is
        // the worse failure.
        if (system) {
          captured.push(system);
          this.granted.system = true;
        } else if (source === "system") {
          throw new CaptureError("no-audio", "that share does not include audio");
        }
      }
    } catch (error) {
      for (const stream of captured) stream.getTracks().forEach((t) => t.stop());
      throw error;
    }

    this.streams.push(...captured);
    await this.buildGraph();
  }

  private async openMicrophone(): Promise<MediaStream> {
    const constraints: MediaTrackConstraints = {
      channelCount: 1,
      // The browser's own processing beats anything we would add, and echo
      // cancellation in particular is required for laptop speakers — but all
      // three are wrong for a room microphone or for music, which is why they
      // are settings rather than constants.
      echoCancellation: this.options.echoCancellation ?? true,
      noiseSuppression: this.options.noiseSuppression ?? true,
      autoGainControl: this.options.autoGainControl ?? true,
    };
    if (this.options.deviceId) constraints.deviceId = { exact: this.options.deviceId };

    try {
      return await navigator.mediaDevices.getUserMedia({ audio: constraints });
    } catch (error) {
      const name = (error as DOMException)?.name;
      if (name === "OverconstrainedError" || name === "NotFoundError") {
        // The remembered device is gone — unplugged, or this is a different
        // machine. Falling back to the default beats refusing to record.
        delete constraints.deviceId;
        return await navigator.mediaDevices.getUserMedia({ audio: constraints });
      }
      if (name === "NotAllowedError" || name === "SecurityError") {
        throw new CaptureError("denied", "microphone permission was refused");
      }
      throw new CaptureError("unavailable", `microphone unavailable: ${name ?? error}`);
    }
  }

  /** Open a shared source, returning null when the share carries no audio. */
  private async openSystemAudio(): Promise<MediaStream | null> {
    if (!canCaptureSystemAudio()) {
      throw new CaptureError("unsupported", "this browser cannot capture system audio");
    }

    let stream: MediaStream;
    try {
      // `video: true` is required for the picker to appear at all, even though
      // the video track is stopped immediately below.
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    } catch (error) {
      const name = (error as DOMException)?.name;
      if (name === "NotAllowedError") {
        throw new CaptureError("denied", "the share was cancelled");
      }
      throw new CaptureError("unavailable", `cannot share audio: ${name ?? error}`);
    }

    for (const track of stream.getVideoTracks()) {
      track.stop();
      stream.removeTrack(track);
    }
    if (stream.getAudioTracks().length === 0) {
      stream.getTracks().forEach((track) => track.stop());
      return null;
    }

    // "Stop sharing", from the browser's own bar, ends the track and tells the
    // page nothing else. Without this the session sits there recording nothing.
    stream.getAudioTracks()[0].addEventListener("ended", () => {
      this.granted.system = false;
      this.options.onSourceEnded?.();
    });
    return stream;
  }

  private async buildGraph(): Promise<void> {
    this.context = new AudioContext();
    const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
    this.moduleUrl = URL.createObjectURL(blob);
    await this.context.audioWorklet.addModule(this.moduleUrl);

    this.node = new AudioWorkletNode(this.context, "pcm-downsampler", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: {
        targetSampleRate: this.options.targetSampleRate ?? 16000,
        frameMs: this.options.frameMs ?? 20,
      },
    });

    this.node.port.onmessage = (event: MessageEvent) => {
      const data = event.data as { type: string; frame?: ArrayBuffer; level?: number };
      if (data.type === "frame" && data.frame) {
        this.options.onFrame(data.frame);
      } else if (data.type === "level" && this.options.onLevel) {
        this.options.onLevel(data.level ?? 0);
      }
    };

    // Every source lands on one gain node, which is both the volume control
    // and the summing point: connecting two sources to the same input is how
    // a Web Audio graph mixes them.
    this.gain = this.context.createGain();
    this.gain.gain.value = this.options.gain ?? 1;
    this.gain.connect(this.node);

    for (const stream of this.streams) {
      const input = this.context.createMediaStreamSource(stream);
      input.connect(this.gain);
      this.inputs.push(input);
    }
  }

  async stop(): Promise<void> {
    // Release in reverse order of acquisition. The MediaStream tracks in
    // particular must be stopped explicitly, or the browser keeps showing the
    // recording indicator — and, for a shared screen, the sharing bar — after
    // the session has ended.
    this.node?.port.close();
    for (const input of this.inputs) input.disconnect();
    this.gain?.disconnect();
    this.node?.disconnect();
    for (const stream of this.streams) stream.getTracks().forEach((track) => track.stop());
    if (this.context) await this.context.close();
    if (this.moduleUrl) URL.revokeObjectURL(this.moduleUrl);

    this.inputs.length = 0;
    this.streams.length = 0;
    this.node = null;
    this.gain = null;
    this.context = null;
    this.moduleUrl = null;
    this.granted = { microphone: false, system: false };
  }
}
