/**
 * UI strings, Japanese and English.
 *
 * Japanese is the default because the product is for a Japanese team; English
 * is the alternate. That ordering is not cosmetic — writing the Japanese first
 * and translating outward keeps the Japanese from reading like a translation,
 * which is the usual outcome when it is bolted onto an English original.
 *
 * The UI language and the *recognition* language are deliberately separate
 * settings. An English-speaking engineer debugging a Japanese meeting needs an
 * English interface over Japanese audio, and collapsing the two into one
 * control makes that impossible.
 */

export type UILang = "ja" | "en";

export interface Strings {
  readonly tagline: string;
  readonly record: string;
  readonly stop: string;
  readonly demo: string;
  readonly demoHint: string;
  readonly status: string;
  readonly backend: string;
  readonly idle: string;
  readonly connecting: string;
  readonly listening: string;
  readonly stopped: string;
  readonly speaking: string;
  readonly silence: string;
  readonly transcript: string;
  readonly minutes: string;
  readonly generate: string;
  readonly generating: string;
  readonly routing: string;
  readonly metrics: string;
  readonly emptyTranscript: string;
  readonly emptyMinutes: string;
  readonly emptyMinutesHint: string;
  readonly participants: string;
  readonly summary: string;
  readonly topics: string;
  readonly decisions: string;
  readonly actions: string;
  readonly owner: string;
  readonly due: string;
  readonly unassigned: string;
  readonly citation: string;
  readonly verified: string;
  readonly dropped: string;
  readonly repairs: string;
  readonly claims: string;
  readonly droppedHint: string;
  readonly audio: string;
  readonly segments: string;
  readonly cost: string;
  readonly frames: string;
  readonly framesDropped: string;
  readonly micDenied: string;
  readonly cannotConnect: string;
  readonly atCapacity: string;
  readonly theme: string;
  readonly uiLanguage: string;
  readonly asrLanguage: string;
  readonly auto: string;
  readonly shortcuts: string;
  readonly toggleRecord: string;
  readonly provider: string;
  readonly score: string;
  readonly errorRate: string;
  readonly costPerMin: string;
  readonly state: string;
  readonly measured: string;
  readonly prior: string;
  readonly chosen: string;
  readonly routingHint: string;
  readonly noSession: string;
  readonly level: string;
  readonly activity: string;
  readonly speakers: string;
}

const JA: Strings = {
  tagline: "日本語・英語 音声AIハーネス",
  record: "録音開始",
  stop: "停止",
  demo: "デモ再生",
  demoHint: "マイクなしでパイプラインの動作を確認できます",
  status: "状態",
  backend: "バックエンド",
  idle: "待機中",
  connecting: "接続中",
  listening: "認識中",
  stopped: "停止",
  speaking: "発話中",
  silence: "無音",
  transcript: "文字起こし",
  minutes: "議事録",
  generate: "議事録を作成",
  generating: "作成中…",
  routing: "ルーティング",
  metrics: "メトリクス",
  emptyTranscript: "録音を開始すると、ここに文字起こしが表示されます。",
  emptyMinutes: "議事録はまだありません。",
  emptyMinutesHint: "録音を停止してから「議事録を作成」を押してください。",
  participants: "出席者",
  summary: "概要",
  topics: "議題",
  decisions: "決定事項",
  actions: "アクションアイテム",
  owner: "担当",
  due: "期限",
  unassigned: "未割当",
  citation: "引用を表示",
  verified: "検証済み",
  dropped: "削除",
  repairs: "修正",
  claims: "件",
  droppedHint: "文字起こしに根拠がないため削除されました",
  audio: "音声",
  segments: "セグメント",
  cost: "コスト",
  frames: "送信フレーム",
  framesDropped: "破棄",
  micDenied: "マイクを利用できません。ブラウザの権限を確認してください。",
  cannotConnect: "サーバーに接続できませんでした。",
  atCapacity: "サーバーが混雑しています。しばらくしてから再試行してください。",
  theme: "テーマ切替",
  uiLanguage: "表示言語",
  asrLanguage: "認識言語",
  auto: "自動判定",
  shortcuts: "ショートカット",
  toggleRecord: "録音の開始・停止",
  provider: "プロバイダ",
  score: "スコア",
  errorRate: "想定エラー率",
  costPerMin: "分あたり",
  state: "状態",
  measured: "実測",
  prior: "推定",
  chosen: "選択",
  routingHint:
    "ルーターは、レイテンシ・コスト・品質の予算に基づいてバックエンドを選択します。スコアは小さいほど良い値です。",
  noSession: "セッションがまだありません。",
  level: "入力レベル",
  activity: "音声区間",
  speakers: "話者",
};

const EN: Strings = {
  tagline: "bilingual JA/EN voice AI harness",
  record: "Record",
  stop: "Stop",
  demo: "Play demo",
  demoHint: "See the pipeline run without a microphone",
  status: "status",
  backend: "backend",
  idle: "idle",
  connecting: "connecting",
  listening: "listening",
  stopped: "stopped",
  speaking: "speech",
  silence: "silence",
  transcript: "Transcript",
  minutes: "Minutes",
  generate: "Generate minutes",
  generating: "Generating…",
  routing: "Routing",
  metrics: "Metrics",
  emptyTranscript: "Start recording and the transcript will appear here.",
  emptyMinutes: "No minutes yet.",
  emptyMinutesHint: "Stop the recording, then press Generate minutes.",
  participants: "Participants",
  summary: "Summary",
  topics: "Topics",
  decisions: "Decisions",
  actions: "Action items",
  owner: "Owner",
  due: "Due",
  unassigned: "unassigned",
  citation: "Show source",
  verified: "verified",
  dropped: "dropped",
  repairs: "repairs",
  claims: "claims",
  droppedHint: "removed — not supported by the transcript",
  audio: "audio",
  segments: "segments",
  cost: "cost",
  frames: "frames sent",
  framesDropped: "dropped",
  micDenied: "Microphone unavailable — check browser permissions.",
  cannotConnect: "Could not connect to the server.",
  atCapacity: "Server is at capacity. Try again shortly.",
  theme: "Toggle theme",
  uiLanguage: "Interface",
  asrLanguage: "Recognition",
  auto: "Auto",
  shortcuts: "Shortcuts",
  toggleRecord: "start / stop recording",
  provider: "provider",
  score: "score",
  errorRate: "expected error",
  costPerMin: "per minute",
  state: "state",
  measured: "measured",
  prior: "prior",
  chosen: "chosen",
  routingHint:
    "The router picks a backend against a latency/cost/quality budget. Lower score is better.",
  noSession: "No session yet.",
  level: "input level",
  activity: "voice activity",
  speakers: "speakers",
};

export function strings(lang: UILang): Strings {
  return lang === "en" ? EN : JA;
}
