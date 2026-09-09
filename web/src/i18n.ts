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
  readonly settings: string;
  readonly providers: string;
  readonly apiKeys: string;
  readonly apiKeysHint: string;
  readonly save: string;
  readonly test: string;
  readonly remove: string;
  readonly close: string;
  readonly testing: string;
  readonly saved: string;
  readonly removed: string;
  readonly notConfigured: string;
  readonly fromEnvironment: string;
  readonly fromEnvironmentHint: string;
  readonly getKey: string;
  readonly keyPlaceholder: string;
  readonly activeModel: string;
  readonly usingMock: string;
  readonly usingMockHint: string;
  readonly storageNote: string;
  readonly statusValid: string;
  readonly statusInvalid: string;
  readonly statusError: string;
  readonly statusUnknown: string;
  readonly search: string;
  readonly searchPlaceholder: string;
  readonly matches: string;
  /** Singular counter; Japanese uses the same counter for both. */
  readonly match: string;
  readonly noMatches: string;
  readonly copy: string;
  readonly copied: string;
  readonly copyFailed: string;
  readonly copyTranscript: string;
  readonly copyMinutes: string;
  readonly shortcutSearch: string;
  readonly shortcutSettings: string;
  readonly shortcutHelp: string;
  readonly shortcutClose: string;
  readonly dismiss: string;
  /* -- workspaces and panels ------------------------------------ */
  readonly wsMeeting: string;
  readonly wsChat: string;
  readonly wsTerminal: string;
  readonly wsCode: string;
  readonly chatPlaceholder: string;
  readonly chatSend: string;
  readonly chatEmpty: string;
  readonly chatThinking: string;
  readonly chatRunning: string;
  readonly chatSteps: string;
  readonly chatToolCall: string;
  readonly chatToolCalls: string;
  readonly terminalPrompt: string;
  readonly terminalStarting: string;
  readonly terminalReady: string;
  readonly terminalStillRunning: string;
  readonly terminalExited: string;
  readonly terminalDisabled: string;
  readonly codeSearch: string;
  readonly codeEmpty: string;
  readonly lines: string;
  readonly truncated: string;
  readonly secPlugins: string;
  readonly pluginsHint: string;
  readonly pluginsDirectory: string;
  readonly pluginsTrust: string;
  readonly pluginReload: string;
  readonly pluginBuiltin: string;
  readonly pluginInactive: string;
  readonly pluginFailed: string;
  readonly toolsProvided: string;
  readonly terminalInteractive: string;
  readonly terminalPlain: string;
  readonly terminalModeHint: string;
  readonly terminalNoPty: string;
  readonly terminalEmulatorFailed: string;
  readonly language: string;
  readonly paletteSearch: string;
  readonly paletteHint: string;
  readonly shortcutWorkspace: string;
  /* -- settings panel ------------------------------------------- */
  readonly secAudio: string;
  readonly secRecognition: string;
  readonly secModels: string;
  readonly secAppearance: string;
  readonly secAbout: string;
  readonly audioSource: string;
  readonly sourceMic: string;
  readonly sourceSystem: string;
  readonly sourceBoth: string;
  readonly sourceMicHint: string;
  readonly sourceSystemHint: string;
  readonly sourceBothHint: string;
  readonly windowsAudioNote: string;
  readonly inputDevice: string;
  readonly systemDefault: string;
  readonly refreshDevices: string;
  readonly deviceNamesHidden: string;
  readonly noInputDevices: string;
  readonly testInput: string;
  readonly stopTest: string;
  readonly testInputHint: string;
  readonly testWhileRecording: string;
  readonly inputGain: string;
  readonly inputGainHint: string;
  readonly processing: string;
  readonly echoCancellation: string;
  readonly noiseSuppression: string;
  readonly autoGainControl: string;
  readonly processingHint: string;
  readonly partialInterval: string;
  readonly partialIntervalHint: string;
  readonly endpointing: string;
  readonly useLanguageDefaults: string;
  readonly useLanguageDefaultsHint: string;
  readonly silenceToEnd: string;
  readonly silenceToEndHint: string;
  readonly speechThreshold: string;
  readonly speechThresholdHint: string;
  readonly themeLabel: string;
  readonly themeSystem: string;
  readonly themeLight: string;
  readonly themeDark: string;
  readonly version: string;
  readonly japaneseTokenizer: string;
  readonly restoreDefaults: string;
  readonly restoreDefaultsHint: string;
  readonly restored: string;
  readonly sourceEnded: string;
  readonly shareCancelled: string;
  readonly shareNoAudio: string;
  readonly shareUnsupported: string;
  /** Server rejection codes, rendered locally. `{label}` / `{prefix}` /
   *  `{env}` are substituted from the response's `context`. */
  readonly errBadPrefix: string;
  readonly errEmptyKey: string;
  readonly errEnvManaged: string;
  readonly sdkMissing: string;
  readonly sdkMissingHint: string;
  /** Verification outcomes, keyed by the server's result code. */
  readonly vNoKey: string;
  readonly vAccepted: string;
  readonly vRejected: string;
  readonly vNoPermission: string;
  readonly vRateLimited: string;
  readonly vUnreachable: string;
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
  shortcuts: "キーボードショートカット",
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
  settings: "設定",
  providers: "プロバイダ",
  apiKeys: "APIキー",
  apiKeysHint: "キーを登録すると、次のリクエストから実際のモデルが使われます。再起動は不要です。",
  save: "保存",
  test: "接続テスト",
  remove: "削除",
  close: "閉じる",
  testing: "テスト中…",
  saved: "保存しました",
  removed: "削除しました",
  notConfigured: "未設定",
  fromEnvironment: "環境変数",
  fromEnvironmentHint: "環境変数が優先されるため、ここからは変更できません。",
  getKey: "キーを取得",
  keyPlaceholder: "キーを貼り付け",
  activeModel: "使用中のモデル",
  usingMock: "モック",
  usingMockHint: "キーが未設定のため、決定論的なモックで動作しています。",
  storageNote:
    "キーはこの端末にのみ保存されます (Windows では DPAPI で暗号化)。表示されるのは末尾4文字のみです。",
  statusValid: "有効",
  statusInvalid: "無効",
  statusError: "エラー",
  statusUnknown: "未確認",
  search: "検索",
  searchPlaceholder: "文字起こしを検索",
  matches: "件",
  match: "件",
  noMatches: "該当なし",
  copy: "コピー",
  copied: "コピーしました",
  copyFailed: "コピーできませんでした",
  copyTranscript: "文字起こしをコピー",
  copyMinutes: "議事録をコピー",
  shortcutSearch: "文字起こしを検索",
  shortcutSettings: "設定を開く",
  shortcutHelp: "このヘルプ",
  shortcutClose: "閉じる / 選択解除",
  dismiss: "閉じる",
  shortcutWorkspace: "ワークスペースの切り替え",
  paletteSearch: "コマンドを検索…",
  paletteHint: "↑↓ で移動 · Enter で実行 · Esc で閉じる",
  terminalInteractive: "対話",
  terminalPlain: "簡易",
  terminalModeHint: "対話モードは本物の端末です。vim や top も動作します。簡易モードは装飾のないテキストで、アシスタントが見るものと同じです。",
  terminalNoPty: "この環境では対話モードを利用できません。pywinpty を導入してください。",
  terminalEmulatorFailed: "ターミナルの読み込みに失敗しました。",
  language: "言語",
  wsMeeting: "議事録",
  wsChat: "チャット",
  wsTerminal: "端末",
  wsCode: "コード",
  chatPlaceholder: "この会議やコードについて質問してください",
  chatSend: "送信",
  chatEmpty: "文字起こし・議事録・ワークスペースのファイルを読めるアシスタントです。ツールを実際に呼び出し、その過程も表示します。",
  chatThinking: "考えています…",
  chatRunning: "実行中",
  chatSteps: "ステップ",
  chatToolCall: "ツール呼び出し",
  chatToolCalls: "ツール呼び出し",
  terminalPrompt: "コマンドを入力",
  terminalStarting: "シェルを起動しています…",
  terminalReady: "シェルを起動しました。cd や環境変数はセッション内で保持されます。",
  terminalStillRunning: "…まだ実行中です。出力を追跡しています。",
  terminalExited: "シェルが終了しました。",
  terminalDisabled: "ターミナルプラグインが無効です。設定 → プラグイン から有効にしてください。",
  codeSearch: "コード内を検索",
  codeEmpty: "左のツリーからファイルを選ぶか、検索してください。表示専用です。",
  lines: "行",
  truncated: "一部のみ",
  secPlugins: "プラグイン",
  pluginsHint: "すべての機能はプラグインとして構成されています。無効にするとアンマウントされ、そのツールも即座に消えます。",
  pluginsDirectory: "プラグインの場所",
  pluginsTrust: "プラグインはサンドボックス化されていません。koe と同じ権限で動作するため、導入前に内容を確認してください。",
  pluginReload: "再読み込み",
  pluginBuiltin: "標準",
  pluginInactive: "停止中",
  pluginFailed: "読み込み失敗",
  toolsProvided: "提供ツール",
  secAudio: "音声",
  secRecognition: "認識",
  secModels: "モデル",
  secAppearance: "表示",
  secAbout: "情報",
  audioSource: "音声の取り込み元",
  sourceMic: "マイク",
  sourceSystem: "画面・ウィンドウの音声",
  sourceBoth: "マイク + 画面音声",
  sourceMicHint: "この端末のマイクから取り込みます。対面の会議向けです。",
  sourceSystemHint: "録音開始時に共有するものを選ぶ画面が表示されます。オンライン会議の相手側の音声を文字起こしする場合はこちらです。",
  sourceBothHint: "自分の声をマイクから、相手の声を画面共有から取り込みます。ハイブリッド会議はこの組み合わせです。",
  windowsAudioNote: "Windows では、画面全体またはブラウザのタブを共有した場合にのみ音声を取り込めます。単一のアプリウィンドウを選ぶと音声は含まれません (Chromium の制限)。共有時は「音声を共有する」を必ず有効にしてください。",
  inputDevice: "入力デバイス",
  systemDefault: "システム既定",
  refreshDevices: "再検出",
  deviceNamesHidden: "デバイス名はマイクの使用許可後に表示されます。",
  noInputDevices: "入力デバイスが見つかりません。",
  testInput: "入力テスト",
  stopTest: "テスト停止",
  testInputHint:
    "選んだ取り込み元で実際に音を拾えているかを確認します。話しかけるとバーが動きます。",
  testWhileRecording: "録音中は入力テストを実行できません。",
  inputGain: "入力ゲイン",
  inputGainHint: "取り込み後に適用する音量です。録音中でも即座に反映されます。",
  processing: "ブラウザ側の音声処理",
  echoCancellation: "エコーキャンセル",
  noiseSuppression: "ノイズ抑制",
  autoGainControl: "自動ゲイン調整",
  processingHint: "通常は有効のままで問題ありません。会議室の集音マイクや音楽を扱う場合は無効にしてください。処理が音声を削ってしまうことがあります。",
  partialInterval: "途中結果の更新間隔",
  partialIntervalHint: "短いほど表示は速くなりますが、その分だけ認識の呼び出し回数が増えます。",
  endpointing: "発話の区切り判定",
  useLanguageDefaults: "言語ごとの既定値を使う",
  useLanguageDefaultsHint: "日本語は英語より長い無音時間を使います。文末の助詞や敬語の前で話者が一呼吸置くため、英語向けの設定では動詞ごと切り落とされます。",
  silenceToEnd: "発話終了と判定する無音時間",
  silenceToEndHint: "実時間の遅延にそのまま加算される値です。短くすると発話が途中で切れます。",
  speechThreshold: "発話と判定する音量",
  speechThresholdHint: "推定したノイズフロアからの相対値です。騒がしい場所では上げてください。",
  themeLabel: "テーマ",
  themeSystem: "システムに合わせる",
  themeLight: "ライト",
  themeDark: "ダーク",
  version: "バージョン",
  japaneseTokenizer: "日本語トークナイザ",
  restoreDefaults: "既定値に戻す",
  restoreDefaultsHint: "この端末の設定のみを初期化します。APIキーは削除されません。",
  restored: "設定を初期化しました。",
  sourceEnded: "画面の共有が終了したため、録音を停止しました。",
  shareCancelled: "共有がキャンセルされました。",
  shareNoAudio: "選択した共有には音声が含まれていません。画面全体またはタブを選び、「音声を共有する」を有効にしてください。",
  shareUnsupported: "このブラウザは画面音声の取り込みに対応していません。",
  errBadPrefix: "{label} のキーは {prefix} で始まります。貼り付けた内容をご確認ください。",
  errEmptyKey: "キーを入力してください。",
  errEnvManaged: "{env} で設定されているため、ここからは変更できません。",
  vNoKey: "キーが設定されていません。",
  vAccepted: "接続できました。",
  vRejected: "キーが拒否されました。",
  vNoPermission: "キーは有効ですが、このモデルへの権限がありません。",
  vRateLimited: "レート制限中ですが、キーは受理されました。",
  vUnreachable: "接続できませんでした。ネットワークをご確認ください。",
  sdkMissing: "ライブラリ未導入",
  sdkMissingHint:
    "このビルドにはクライアントライブラリが含まれていないため、キーがあってもモックのままです。pip install 'koe-harness[llm]' を実行してください。",
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
  shortcuts: "Keyboard shortcuts",
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
  settings: "Settings",
  providers: "Providers",
  apiKeys: "API keys",
  apiKeysHint: "Add a key and the real model is used from your next request. No restart needed.",
  save: "Save",
  test: "Test",
  remove: "Remove",
  close: "Close",
  testing: "Testing…",
  saved: "Saved",
  removed: "Removed",
  notConfigured: "not configured",
  fromEnvironment: "environment",
  fromEnvironmentHint: "Set by an environment variable, which takes precedence, so it cannot be edited here.",
  getKey: "Get a key",
  keyPlaceholder: "Paste key",
  activeModel: "Active model",
  usingMock: "mock",
  usingMockHint: "No key configured, so deterministic mocks are in use.",
  storageNote:
    "Keys are stored on this machine only (encrypted with DPAPI on Windows). Only the last four characters are ever shown.",
  statusValid: "valid",
  statusInvalid: "invalid",
  statusError: "error",
  statusUnknown: "unverified",
  search: "Search",
  searchPlaceholder: "Search the transcript",
  matches: "matches",
  match: "match",
  noMatches: "no matches",
  copy: "Copy",
  copied: "Copied",
  copyFailed: "Could not copy",
  copyTranscript: "Copy transcript",
  copyMinutes: "Copy minutes",
  shortcutSearch: "search the transcript",
  shortcutSettings: "open settings",
  shortcutHelp: "this help",
  shortcutClose: "close / clear selection",
  dismiss: "Dismiss",
  shortcutWorkspace: "switch workspace",
  paletteSearch: "Search commands…",
  paletteHint: "↑↓ to move · Enter to run · Esc to close",
  terminalInteractive: "Interactive",
  terminalPlain: "Plain",
  terminalModeHint: "Interactive is a real terminal — vim and top work in it. Plain is undecorated text, and is what the assistant sees.",
  terminalNoPty: "Interactive mode is unavailable here. Install pywinpty to enable it.",
  terminalEmulatorFailed: "The terminal emulator failed to load.",
  language: "Language",
  wsMeeting: "Meeting",
  wsChat: "Chat",
  wsTerminal: "Terminal",
  wsCode: "Code",
  chatPlaceholder: "Ask about this meeting, or about the code",
  chatSend: "Send",
  chatEmpty: "An assistant that can read the transcript, the 議事録, and this workspace's files. It calls tools for real, and shows you the calls.",
  chatThinking: "Working…",
  chatRunning: "running",
  chatSteps: "steps",
  chatToolCall: "tool call",
  chatToolCalls: "tool calls",
  terminalPrompt: "Type a command",
  terminalStarting: "Starting a shell…",
  terminalReady: "Shell ready. cd and environment variables persist across commands.",
  terminalStillRunning: "…still running; following the output.",
  terminalExited: "The shell exited.",
  terminalDisabled: "The terminal plugin is disabled. Enable it in Settings → Plugins.",
  codeSearch: "Search the code",
  codeEmpty: "Pick a file from the tree, or search. This view is read-only.",
  lines: "lines",
  truncated: "truncated",
  secPlugins: "Plugins",
  pluginsHint: "Every capability is a plugin. Disabling one unmounts it — its tools disappear immediately, they do not merely stop responding.",
  pluginsDirectory: "Plugins directory",
  pluginsTrust: "Plugins are not sandboxed. They run with the same privileges as koe, so read one before you install it.",
  pluginReload: "Reload",
  pluginBuiltin: "built-in",
  pluginInactive: "inactive",
  pluginFailed: "failed to load",
  toolsProvided: "tools",
  secAudio: "Audio",
  secRecognition: "Recognition",
  secModels: "Models",
  secAppearance: "Appearance",
  secAbout: "About",
  audioSource: "Audio source",
  sourceMic: "Microphone",
  sourceSystem: "Screen or window audio",
  sourceBoth: "Microphone + screen audio",
  sourceMicHint: "Capture from this machine's microphone. For an in-person meeting.",
  sourceSystemHint: "You pick what to share when recording starts. This is how you transcribe the far side of an online meeting.",
  sourceBothHint: "Your voice from the microphone, everyone else from the shared audio. This is the combination a hybrid meeting needs.",
  windowsAudioNote: "On Windows, audio is only available when you share an entire screen or a browser tab — a single application window carries no audio (a Chromium limitation). Tick “Share audio” in the picker.",
  inputDevice: "Input device",
  systemDefault: "System default",
  refreshDevices: "Rescan",
  deviceNamesHidden: "Device names appear once microphone permission has been granted.",
  noInputDevices: "No input devices found.",
  testInput: "Test input",
  stopTest: "Stop test",
  testInputHint:
    "Check that the source you picked is actually picking anything up. Speak, and the bar moves.",
  testWhileRecording: "The input test cannot run while recording.",
  inputGain: "Input gain",
  inputGainHint: "Applied after capture. Takes effect immediately, even while recording.",
  processing: "Browser audio processing",
  echoCancellation: "Echo cancellation",
  noiseSuppression: "Noise suppression",
  autoGainControl: "Automatic gain control",
  processingHint: "Leave these on for a normal setup. Turn them off for a room microphone or for music, where the processing removes signal.",
  partialInterval: "Interim update interval",
  partialIntervalHint: "Lower feels faster and costs proportionally more: every interim result is another decode of the utterance so far.",
  endpointing: "Endpointing",
  useLanguageDefaults: "Use the language defaults",
  useLanguageDefaultsHint: "Japanese uses a longer silence window than English: speakers pause before sentence-final particles and polite endings, and an English-tuned endpointer cuts there — removing the verb.",
  silenceToEnd: "Silence before an utterance ends",
  silenceToEndHint: "Added directly to every utterance's latency. Shortening it truncates speech.",
  speechThreshold: "Speech threshold",
  speechThresholdHint: "Measured against the tracked noise floor. Raise it in a noisy room.",
  themeLabel: "Theme",
  themeSystem: "Match system",
  themeLight: "Light",
  themeDark: "Dark",
  version: "Version",
  japaneseTokenizer: "Japanese tokenizer",
  restoreDefaults: "Restore defaults",
  restoreDefaultsHint: "Resets this device's preferences only. API keys are not removed.",
  restored: "Preferences restored.",
  sourceEnded: "Screen sharing ended, so recording stopped.",
  shareCancelled: "The share was cancelled.",
  shareNoAudio: "That share carries no audio. Pick an entire screen or a tab, and tick “Share audio”.",
  shareUnsupported: "This browser cannot capture screen audio.",
  errBadPrefix: "{label} keys start with {prefix} — check what you pasted.",
  errEmptyKey: "Enter a key.",
  errEnvManaged: "Set by {env}, so it cannot be changed here.",
  vNoKey: "No key configured.",
  vAccepted: "Connected.",
  vRejected: "The key was rejected.",
  vNoPermission: "The key is valid but lacks permission for this model.",
  vRateLimited: "Rate limited, but the key is accepted.",
  vUnreachable: "Could not reach the provider — check your network.",
  sdkMissing: "library missing",
  sdkMissingHint:
    "This build has no client library for it, so the mock stays in use even with a key. Install it with pip install 'koe-harness[llm]'.",
};

export function strings(lang: UILang): Strings {
  return lang === "en" ? EN : JA;
}
