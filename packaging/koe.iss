; Inno Setup script for the koe desktop application.
;
; Compiled by packaging/build.py --installer. Produces
; dist/koe-setup-<version>.exe.
;
; Two choices worth stating:
;
;   * **Per-user install by default** (PrivilegesRequired=lowest). koe needs no
;     machine-wide component, and a per-user install avoids the UAC prompt that
;     makes people abandon an install of a tool they were merely curious about.
;     It also means the app can update itself later without elevation.
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

; Per-user: no elevation, no UAC prompt.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
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

[Code]
function WebView2Installed: Boolean;
var
  Version: String;
begin
  // The runtime registers under either hive depending on per-machine or
  // per-user install, so both are checked.
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
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
