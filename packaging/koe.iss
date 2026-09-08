; Inno Setup script for the koe desktop application.
;
; Compiled by packaging/build.py --installer. Produces
; dist/koe-setup-<version>.exe.
;
; Three choices worth stating:
;
;   * **Per-user install, and no question about it** (PrivilegesRequired=lowest).
;     koe needs no machine-wide component, so a per-user install avoids the UAC
;     prompt that makes people abandon an install of a tool they were merely
;     curious about, and lets the app update itself later without elevation.
;     The "install for me / for all users" chooser is deliberately *not*
;     offered: it is the first thing a user sees, it asks a question they have
;     no basis to answer, and both answers lead to the same working app.
;
;   * **The wizard language follows Windows.** ShowLanguageDialog=auto picks
;     Japanese on a Japanese system and English on an English one, and only
;     asks when it is neither. A language prompt before the installer has said
;     anything is a dialog spent on a question the OS already answered.
;
;   * **The WebView2 runtime is checked, not bundled.** It ships with Windows 11
;     and with updated Windows 10, so bundling the installer would add weight
;     for almost everyone to benefit almost nobody. If it is genuinely missing,
;     the user is told where to get it rather than met with a blank window.

#define AppName "koe"
#define AppVersion "0.1.0"
#define AppPublisher "jon-jc"
#define AppURL "https://github.com/jon-jc/koe-harness"
#define AppExe "koe.exe"

[Setup]
AppId={{8C4E1A6F-2D93-4F5B-9E77-1A2B3C4D5E6F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
VersionInfoVersion={#AppVersion}

; Per-user: no elevation, no UAC prompt, and no chooser dialog.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes

OutputDir=..\dist
OutputBaseFilename=koe-setup-{#AppVersion}
SetupIconFile=assets\koe.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}

; LZMA2/max: the payload is mostly the Japanese dictionary, which compresses
; well, and an installer is downloaded far more often than it is built.
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

LicenseFile=..\LICENSE
MinVersion=10.0

; Follow the OS; ask only when it is neither Japanese nor English.
ShowLanguageDialog=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[CustomMessages]
english.CreateDesktopIcon=Create a &desktop shortcut
english.LaunchApp=Launch {#AppName}
english.WebView2Missing=Microsoft Edge WebView2 Runtime is required and was not found.%n%nkoe will install, but will not start until the runtime is present.%n%nDownload it from:%nhttps://developer.microsoft.com/microsoft-edge/webview2/
japanese.CreateDesktopIcon=デスクトップにショートカットを作成する(&D)
japanese.LaunchApp={#AppName} を起動する
japanese.WebView2Missing=Microsoft Edge WebView2 ランタイムが見つかりませんでした。%n%nインストールは続行できますが、ランタイムがないと koe は起動しません。%n%n次の場所から入手してください:%nhttps://developer.microsoft.com/microsoft-edge/webview2/

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller output directory.
Source: "..\dist\koe\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Logs and the webview cache are generated at runtime and are not tracked by
; the installer, so they would otherwise be left behind. Settings and any saved
; meetings are deliberately preserved — an uninstall should not silently
; destroy a user's meeting records.
Type: filesandordirs; Name: "{localappdata}\koe\logs"
Type: filesandordirs; Name: "{localappdata}\koe\webview"
; The single-instance lock names a pid that will not exist after uninstall.
; Harmless if left — the app clears a stale one — but leaving a file behind
; that only we understand is not a clean uninstall.
Type: files; Name: "{localappdata}\koe\koe.lock"

[Code]
function WebView2Installed: Boolean;
var
  Version: String;
begin
  // The runtime registers under a different key depending on per-machine
  // versus per-user install and on registry redirection, so all three are
  // checked before falling back to looking for Edge itself.
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    FileExists(ExpandConstant('{pf32}\Microsoft\Edge\Application\msedge.exe')) or
    FileExists(ExpandConstant('{pf}\Microsoft\Edge\Application\msedge.exe'));
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  if not WebView2Installed then
    // Warn rather than block: the runtime can be installed afterwards, and
    // refusing the install outright would be more annoying than useful.
    MsgBox(ExpandConstant('{cm:WebView2Missing}'), mbInformation, MB_OK);
end;
