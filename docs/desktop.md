# The desktop application

A native window over the same FastAPI application the server deployment runs,
on loopback. **Nothing is stubbed for desktop** — the window is a client of the
real API, which is what stops the two builds from drifting apart.

```bash
pip install -e ".[api,cli,ja,desktop]"
python -m koe.desktop          # or: koe-desktop
```

## Building the installer

```bash
pip install -e ".[api,llm,ja,cli,desktop,build]"
python packaging/build.py --installer
```

Produces:

| Artifact | Size |
|---|---|
| `dist/koe/` (portable folder) | 293 MB |
| `dist/koe-setup-<version>.exe` | 47 MB |

Flags: `--no-japanese` (drops MeCab, ~248 MB smaller), `--skip-web` (use the
committed bundle), `--skip-verify`, `--clean`. `--version X.Y.Z` stamps a
release version into the bundle, and `--manifest` writes `dist/latest.json`,
the file installed copies read to find a release (see [Updates](#updates)).

CI builds this on `windows-latest` on every push and uploads the installer as
an artifact, so the build is proven on a clean machine rather than only on the
one it was written on. Every merge to `main` also publishes it as a release,
which is what installed copies update from.

---

## Why a webview

| Option | Why not |
|---|---|
| **Electron** | Ships a second copy of Chromium (~150 MB) to run a 38 KB client, on a platform that already has WebView2 |
| **Tauri** | Smaller, but adds a Rust toolchain to a project that is Python and TypeScript |
| **A native toolkit** (Qt/wx) | A second implementation of the transcript view — precisely the thing most likely to diverge. And `AudioWorklet`, which the capture path depends on, does not exist outside a browser engine |

pywebview over WebView2 reuses the existing client, adds no second runtime, and
keeps the capture path identical to the browser build. The cost is a dependency
on the system WebView2 runtime, which ships with Windows 11 and updated Windows
10 — checked at startup and reported as a clear message rather than a blank
window.

## Size

293 MB unpacked, 47 MB compressed. **248 MB of that is `unidic-lite`**, the
Japanese morphological dictionary, and it is the entire reason for the
`--no-japanese` flag.

Including it is the default because Japanese is the product's premise. koe
degrades honestly without it — falling back to character segmentation and
*labelling* that fallback wherever a number depends on it (see
`tests/text/test_backend_parity.py`) — so the slim build is defensible, just
not the right default for this audience.

---

## Production concerns, and how each is handled

### Startup

**The port is bound before the server starts.** The app asks the OS for port 0
and reads back the assignment, so the window knows its URL with certainty. The
alternative — guess a port, poll until something answers — has a startup race
*and* a conflict whenever the guess is taken, and both failures look identical
to the user: "it didn't open".

**Startup is awaited, not slept on.** The window does not load until uvicorn
reports the application lifespan complete.

### Single instance

Double-clicking an icon twice is a normal thing to do. Two copies would each
start a server, each hold the microphone, and each write the same settings
file.

The lock is a PID file. The subtlety is **stale locks**: after a crash the file
remains, and a naive implementation then refuses to ever start again. A lock
whose PID is no longer alive is taken over. A PID can be recycled, so this can
theoretically report a stale lock as live — the failure mode is a spurious
"already running" rather than two instances corrupting each other, which is the
right way round.

### Shutdown

Closing the window saves geometry, then asks uvicorn to stop and gives it up to
10 seconds to drain. Past the deadline the process exits anyway: **a GUI that
will not close is worse than one that drops a session.**

### Settings

Window geometry, theme and language persist to `%APPDATA%\koe\settings.json`,
written atomically via a temporary file and a replace — a crash partway through
a direct write leaves truncated JSON, and the next launch would then lose
*every* preference rather than the one being changed.

Every read is defensive: truncated files, non-object JSON, unknown keys from a
newer version, and wrong types all fall back to defaults rather than stopping
the app. A window restored onto a monitor that has since been unplugged is
re-centred instead of opening off-screen, where it is running but invisible and
the user concludes it failed to start.

### Logging

`%LOCALAPPDATA%\koe\logs\koe.log`, rotating at 2 MB × 3. Structured JSON, the
same format the server writes.

This matters more on desktop than on a server: a frozen GUI app has no console,
so `stderr` goes nowhere, and without a file a crash report is "it closed" and
nothing else.

### Crash reporting

`sys.excepthook` and `threading.excepthook` log the traceback and show a message
box naming the log directory. A windowed application that exits silently is
indistinguishable from one that never launched.

### Capability probe

On load, the app inspects what the embedded browser can actually do and logs it:

```json
{"message": "webview capabilities", "secureContext": true, "mediaDevices": true,
 "audioWorklet": true, "webSocket": true, "userAgent": "...Edg/152.0.0.0"}
```

Support for a desktop app is somebody describing a symptom over email, so the
log has to answer the first question — *could it even reach the microphone?* —
without a round trip. It inspects capability only; it deliberately does not call
`getUserMedia`, which would switch the microphone on without the user asking.

### Microphone permission

WebView2 treats an unhandled permission request conservatively, and pywebview
registers no handler — so without intervention, pressing Record produces a
silent denial.

The app sets `--use-fake-ui-for-media-stream` on the embedded browser only. This
is a genuine trade, stated plainly: the app cannot render its own permission
prompt, so it grants capture to its own loopback origin and relies on the
recording state being unmistakable in the UI. It does not affect the user's
actual browser.

### Headless mode

`KOE_DESKTOP_HEADLESS=1` starts the server without a window and writes the port
to `KOE_DESKTOP_PORT_FILE`. This is what `packaging/build.py` uses to verify a
frozen build, and what lets CI test the artifact where there is no display.

---

## Verified

Every claim below was checked against the built artifact, not assumed:

```
exe properties     ProductName koe · FileVersion 0.1.0 · publisher jon-jc
frozen launch      health ok · japanese_tokenizer mecab
frozen pipeline    6 segments via mock-accurate, speaker 田中
installer (silent) /VERYSILENT → 299 MB, koe.exe + uninstaller present
installer (wizard) driven through every page → installed to
                   %LOCALAPPDATA%\Programs\koe, Start Menu shortcuts created,
                   "Launch koe" opened the window and bound a port
installed binary   /health ok, /v1/transcribe returns 6 JA segments
uninstall          clean; settings and meetings deliberately preserved
webview            secureContext, mediaDevices, audioWorklet all true
```

The two installer rows are separate because for a while only the first existed,
and it hid a real problem. `/VERYSILENT` skips the language prompt, the licence
page, the install-mode chooser and the post-install launch — so a silent install
passing says nothing about what a person double-clicking the file experiences.
Driving the wizard is what surfaced two dialogs that were asking questions
nobody needed to answer.

### The download is the part that fails

A locally built installer runs. A *downloaded* one does not, and that
distinction is invisible until you test it:

```
copy of installer + Mark of the Web (ZoneId=3)
  → double-click → smartscreen.exe runs, setup process exits, no window
```

That is what an unsigned binary does when it arrives from a browser. Nothing
about the installer is wrong; SmartScreen simply has no reputation for the file.
The mitigations that are actually available without a certificate are all in
place: releases publish the raw `.exe` rather than a login-gated zip, a
`SHA256SUMS.txt` ships beside it so the download can be verified, both READMEs
state exactly what Windows will show and what to click, and a `pip install`
route exists for anyone who would rather not click through a security warning at
all.

## Updates

Every merge to `main` publishes a release (`.github/workflows/release.yml`), and
an installed koe keeps itself current from those releases.

```
merge to main
  → release workflow: build, verify the frozen exe, compile the installer
  → publish v0.1.<commits on main>: koe-setup-<v>.exe, SHA256SUMS.txt, latest.json

installed koe (20 s after launch, then every 6 h)
  → releases/latest → latest.json → newer than this build?
  → download in the background, hashing as it streams
  → SHA-256 and size must match latest.json, or the file is deleted
  → "Update ready · Restart" appears in the sidebar
  → on close, or Restart: the installer runs silently, waits for koe's PID
    to exit, replaces the files, and (for Restart) starts koe again
```

| Decision | Why |
|---|---|
| **The version is the commit count** | Pull requests are squash-merged, so each merge adds exactly one commit: the number only rises, and it is reproducible from the repository without a counter stored anywhere |
| **Verified while downloading, and again at launch** | The file sits in a user-writable directory between the check and the click, so the digest is re-checked immediately before the installer starts |
| **Only this repository's release URLs** | `latest.json` names the digest but cannot redirect the download: an installer URL that is not `github.com/jon-jc/koe-harness/releases/download/<tag>/<installer>` is refused, as are `..`, query strings and percent-encoding |
| **Installs on close, not mid-meeting** | The download is invisible; the restart is the person's choice or happens when they close the app anyway, and Restart is refused while a meeting is recording |
| **The installer waits for koe to exit** | It is started with `/waitpid=<pid>`, so nothing is replaced while the old process still holds its files. Restart Manager (`CloseApplications`) is the backstop |
| **Cross-origin requests are refused** | The update routes are on the loopback server, which any page in a real browser can reach; a request carrying a foreign `Origin` cannot start an install |
| **Failure is a status, not a dialog** | No network, GitHub's rate limit, a bad digest: logged, shown in Settings → About, and retried at the next check |

Turn it off in **Settings → About → Install updates automatically**. With it
off, nothing is checked in the background; **Check now** still downloads and
verifies, and the update installs only when you press **Restart to update**.

`KOE_UPDATE_FEED` points an installed build at a different `latest.json` —
HTTPS, or plain HTTP on loopback only — which is how an update is exercised
before it is published. Headless mode (`KOE_DESKTOP_HEADLESS`) never checks in
the background, so the packaging step and CI do not download releases while
verifying a build.

What it cannot do: a copy installed before this feature has no updater and has
to be replaced by hand once. And the installer is still unsigned — the digest
proves the file is the one the release workflow published, not who ran the
workflow.

## Not done

- **The binary is unsigned.** SmartScreen blocks a downloaded copy behind
  *More info → Run anyway*. Code signing needs a certificate — an OV
  certificate is a recurring cost, and even a fresh one carries no reputation
  for weeks. Without it, signing is not something that can be faked
  convincingly, and pretending otherwise would be worse than saying so. What
  the project does instead is documented above: publish the hash, say what the
  warning means, and offer a route that avoids it.
- **Windows only.** The packaging is Windows-specific. The application code is
  not — `koe.desktop.paths` already resolves macOS and XDG locations — but no
  `.app` or `.AppImage` is produced.
