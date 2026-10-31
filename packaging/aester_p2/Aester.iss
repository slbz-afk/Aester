#ifndef StagePath
  #error StagePath is required
#endif
#ifndef VersionId
  #error VersionId is required
#endif
#ifndef BuildOutput
  #error BuildOutput is required
#endif
#ifndef AppIdentity
  #define AppIdentity "BD0B9C56-1F68-4D33-9C51-208037FC67B7"
#endif
#ifndef InstallerName
  #define InstallerName "Aester-P2-" + VersionId + "-Setup"
#endif
#ifndef AppMB
  #define AppMB "60"
#endif
; Disk use of each component after installation, measured 2026-09-27 on a real
; installation (installed files plus the kept download cache). Inno otherwise
; counts only the ~60 MB application files. See docs/INSTALLER_UX_FIX_*_20260927.md.
#define BaseExtraMB 1060
#define FiniteExtraMB 720
#define GimpExtraMB 802
#define SkillExtraMB 1

[Setup]
AppId={{{#AppIdentity}}
AppName=Aester
AppVersion=0.2.0.24
AppVerName=Aester P2 Preview
AppPublisher=Aester Project
DefaultDirName={localappdata}\Programs\Aester
DefaultGroupName=Aester
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern dynamic
WizardSizePercent=115
SetupIconFile={#StagePath}\Aester.ico
UninstallDisplayIcon={app}\Aester.exe
UninstallFilesDir={app}\uninstall
OutputDir={#BuildOutput}
OutputBaseFilename={#InstallerName}
Compression=lzma2/fast
SolidCompression=no
DisableProgramGroupPage=yes
InfoBeforeFile={#StagePath}\INSTALL-NOTES.txt
CloseApplications=no
RestartApplications=no
ShowLanguageDialog=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"; InfoBeforeFile: "{#StagePath}\INSTALL-NOTES-en.txt"; LicenseFile: "{#StagePath}\LICENSE-PAGE.en.txt"
Name: "zh_CN"; MessagesFile: "ChineseSimplified.isl"; LicenseFile: "{#StagePath}\LICENSE-PAGE.zh_CN.txt"

#include "installer-messages.iss"

[Types]
Name: "basic"; Description: "{cm:TypeBasic}"
Name: "full"; Description: "{cm:TypeFull}"
Name: "custom"; Description: "{cm:TypeCustom}"; Flags: iscustom

[Components]
Name: "base"; Description: "{cm:ComponentBase}"; Types: basic full custom; Flags: fixed; ExtraDiskSpaceRequired: {#BaseExtraMB * 1048576}
#ifdef IncludeFinite
Name: "finite"; Description: "{cm:ComponentFinite}"; Types: full; ExtraDiskSpaceRequired: {#FiniteExtraMB * 1048576}
#endif
#ifdef IncludeRetouch
Name: "gimp"; Description: "{cm:ComponentGimp}"; Types: full; ExtraDiskSpaceRequired: {#GimpExtraMB * 1048576}
Name: "retouch_skill"; Description: "{cm:ComponentRetouchSkill}"; Types: full; ExtraDiskSpaceRequired: {#SkillExtraMB * 1048576}
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; Flags: unchecked

[Files]
; Only these temporary prerequisites are extracted before the application commits.
Source: "{#StagePath}\bootstrap\*"; DestDir: "{tmp}\aester-bootstrap"; Flags: dontcopy recursesubdirs
#ifdef IncludeFinite
Source: "{#StagePath}\finite-bundle\*"; DestDir: "{tmp}\aester-finite"; Flags: dontcopy recursesubdirs
#endif
Source: "{#StagePath}\application\*"; DestDir: "{app}\versions\{#VersionId}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#StagePath}\Aester.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagePath}\Aester-MCP.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagePath}\Aester.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagePath}\uninstall-support\*"; DestDir: "{app}\uninstall-support"; Flags: ignoreversion

[Icons]
#ifndef NoIcons
Name: "{autoprograms}\Aester"; Filename: "{app}\Aester.exe"; IconFilename: "{app}\Aester.ico"
Name: "{autodesktop}\Aester"; Filename: "{app}\Aester.exe"; Tasks: desktopicon
#endif

[Run]
Filename: "{app}\Aester.exe"; Description: "{cm:Launch}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Only after the guarded helper succeeded. Never delete user paths recursively.
Type: files; Name: "{app}\install-state.json"
Type: files; Name: "{app}\aester-installation.json"
Type: files; Name: "{app}\lifecycle.json"
Type: files; Name: "{app}\runtime-use.lock"
Type: files; Name: "{app}\uninstall.pending"
Type: files; Name: "{app}\uninstall-result.json"
Type: files; Name: "{app}\uninstall-report.json"
Type: files; Name: "{app}\uninstall-report.txt"
Type: files; Name: "{app}\install-language.txt"
; Hash handovers (starstack.hashing.handover_path) a stopped workflow worker never read; only
; Aester's own folder in the user's temporary folder, not recursive.
Type: files; Name: "{%TEMP}\aester-hash-handover\*.json"
Type: files; Name: "{%TEMP}\aester-hash-handover\*.tmp"
Type: dirifempty; Name: "{%TEMP}\aester-hash-handover"

[Code]
var
  ProgressPage: TOutputProgressWizardPage;
  ComponentRoot, DownloadCache, DownloadSource, WorkerParams, ResultFile: String;
  GimpCaptionDone, GimpPresent: Boolean;
  GimpIndex: Integer;
  PreviousComponentsClick, PreviousTypesChange: TNotifyEvent;
  MaintenanceMutex: THandle;

function WinCreateMutex(Security: Cardinal; InitialOwner: Boolean; Name: String): THandle;
  external 'CreateMutexW@kernel32.dll stdcall';
function WinCloseHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';
function AcquireMaintenance: Boolean;
var Name: String; Error: Cardinal;
begin
  Name := 'Global\AesterMaintenance-{#AppIdentity}-' +
    GetSHA256OfString(UTF8Encode(Lowercase(ExpandConstant('{localappdata}'))));
  MaintenanceMutex := WinCreateMutex(0, False, Name);
  Error := DLLGetLastError;
  Result := (MaintenanceMutex <> 0) and (Error <> 183);
  if not Result then begin
    if MaintenanceMutex <> 0 then WinCloseHandle(MaintenanceMutex);
    MaintenanceMutex := 0;
    SuppressibleMsgBox(CustomMessage('MaintenanceBusy'), mbError, MB_OK, IDOK);
  end;
end;

function InitializeSetup: Boolean;
begin
  Result := AcquireMaintenance;
end;

function InitializeUninstall: Boolean;
begin
  Result := AcquireMaintenance;
end;

procedure DeinitializeSetup;
var Code: Integer;
begin
  if (WorkerParams <> '') and FileExists(ExpandConstant('{tmp}\aester-bootstrap\python.exe')) then
    Exec(ExpandConstant('{tmp}\aester-bootstrap\python.exe'), WorkerParams + ' --rollback',
      ExpandConstant('{tmp}\aester-bootstrap'), SW_HIDE, ewWaitUntilTerminated, Code);
  if MaintenanceMutex <> 0 then WinCloseHandle(MaintenanceMutex);
end;

procedure DeinitializeUninstall;
begin
  if MaintenanceMutex <> 0 then WinCloseHandle(MaintenanceMutex);
end;

#include AddBackslash(StagePath) + "uninstall-files.iss"
#include "uninstall-ui.iss"

function Q(const S: String): String;
begin
  Result := '"' + S + '"';
end;

#include "worker-runner.iss"

procedure InitializeWizard;
begin
  { A first installation starts from the basic type. An upgrade keeps the type and
    components Inno restored from last time; forcing "basic" here showed "basic"
    above a fully ticked list (user report, 2026-09-28). }
  if not RegValueExists(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{' + '{#AppIdentity}' + '}_is1',
      'Inno Setup: Setup Type') then
    WizardForm.TypesCombo.ItemIndex := 0;
  WizardForm.ComponentsDiskSpaceLabel.WordWrap := True;
  ProgressPage := CreateOutputProgressPage(CustomMessage('PrepareTitle'), CustomMessage('PrepareDescription'));
end;

procedure ShowPreparationNotices;
var Raw: AnsiString; Text: String;
begin
  if not LoadStringFromFile(ChangeFileExt(ResultFile, '.notices.txt'), Raw) then exit;
  Text := Trim(UTF8Decode(Raw));
  if Text = '' then exit;
  Log('Aester preparation notices: ' + Text);
  if not WizardSilent then MsgBox(Text, mbInformation, MB_OK);
end;

{ --------------------------- existing GIMP, shown before anything is chosen }

function ShortVersion(const Value: String): String;
var I, Dots: Integer;
begin
  Result := '';
  Dots := 0;
  for I := 1 to Length(Value) do begin
    if Value[I] = '.' then begin
      Dots := Dots + 1;
      if Dots = 3 then exit;
    end;
    Result := Result + Value[I];
  end;
end;

procedure CheckGimpKey(const Root: Integer; const Name: String; var Found: Boolean; var Version: String);
var Key, Value: String;
begin
  Key := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + Name;
  if RegKeyExists(Root, Key) then begin
    Found := True;
    if (Version = '') and RegQueryStringValue(Root, Key, 'DisplayVersion', Value) then
      Version := ShortVersion(Trim(Value));
  end;
end;

function ExistingGimp(var Version: String): Boolean;
begin
  Result := False;
  Version := '';
  CheckGimpKey(HKCU, 'GIMP-3_is1', Result, Version);
  CheckGimpKey(HKLM64, 'GIMP-3_is1', Result, Version);
  CheckGimpKey(HKLM32, 'GIMP-3_is1', Result, Version);
  CheckGimpKey(HKCU, 'GIMP-2_is1', Result, Version);
  CheckGimpKey(HKLM64, 'GIMP-2_is1', Result, Version);
  CheckGimpKey(HKLM32, 'GIMP-2_is1', Result, Version);
  Result := Result or (GetEnv('AESTER_GIMP') <> '') or
    FileExists(ExpandConstant('{localappdata}\Programs\GIMP 3\bin\gimp-console-3.2.exe')) or
    FileExists(ExpandConstant('{commonpf64}\GIMP 3\bin\gimp-console-3.2.exe')) or
    FileExists(ExpandConstant('{commonpf32}\GIMP 3\bin\gimp-console-3.2.exe'));
end;

{ Our own disk estimate: the application files plus every selected component,
  leaving out a GIMP that is already installed (it is reused, never reinstalled). }
procedure UpdateDiskSpaceLabel;
var MB: Extended;
begin
  { Inno refreshes the size column on every click; an existing GIMP is reused, not installed. }
  if GimpPresent and (GimpIndex >= 0) then WizardForm.ComponentsList.ItemSubItem[GimpIndex] := '';
  MB := {#AppMB} + {#BaseExtraMB};
#ifdef IncludeFinite
  if WizardIsComponentSelected('finite') then MB := MB + {#FiniteExtraMB};
#endif
#ifdef IncludeRetouch
  if WizardIsComponentSelected('gimp') and not GimpPresent then MB := MB + {#GimpExtraMB};
  if WizardIsComponentSelected('retouch_skill') then MB := MB + {#SkillExtraMB};
  if WizardIsComponentSelected('gimp') and GimpPresent then
    WizardForm.ComponentsDiskSpaceLabel.Caption :=
      FmtMessage(CustomMessage('DiskSpaceEstimateGimpReused'), [Format('%.1f', [MB / 1024])])
  else
#endif
    WizardForm.ComponentsDiskSpaceLabel.Caption :=
      FmtMessage(CustomMessage('DiskSpaceEstimate'), [Format('%.1f', [MB / 1024])]);
end;

procedure ComponentsClickCheck(Sender: TObject);
begin
  PreviousComponentsClick(Sender);
  UpdateDiskSpaceLabel;
end;

procedure TypesComboChange(Sender: TObject);
begin
  PreviousTypesChange(Sender);
  UpdateDiskSpaceLabel;
end;

procedure CurPageChanged(CurPageID: Integer);
var I: Integer; Version, Caption: String;
begin
  if (CurPageID <> wpSelectComponents) or GimpCaptionDone then begin
    if CurPageID = wpSelectComponents then UpdateDiskSpaceLabel;
    exit;
  end;
  GimpCaptionDone := True;
  GimpIndex := -1;
  PreviousComponentsClick := WizardForm.ComponentsList.OnClickCheck;
  WizardForm.ComponentsList.OnClickCheck := @ComponentsClickCheck;
  PreviousTypesChange := WizardForm.TypesCombo.OnChange;
  WizardForm.TypesCombo.OnChange := @TypesComboChange;
#ifdef IncludeRetouch
  GimpPresent := ExistingGimp(Version);
  if GimpPresent then begin
    if Version = '3.2.6' then Caption := CustomMessage('GimpFoundSame')
    else if Version <> '' then Caption := FmtMessage(CustomMessage('GimpFoundOther'), [Version])
    else Caption := CustomMessage('GimpFoundUnknown');
    for I := 0 to WizardForm.ComponentsList.Items.Count - 1 do
      if WizardForm.ComponentsList.ItemCaption[I] = CustomMessage('ComponentGimp') then begin
        WizardForm.ComponentsList.ItemCaption[I] := Caption;
        GimpIndex := I;
      end;
  end;
#endif
  UpdateDiskSpaceLabel;
end;

procedure ProgressLog(const S: String; const Error, FirstLine: Boolean);
var Tail, Number: String; Split: Integer;
begin
  Log(S);
  if Copy(S, 1, 9) = 'PROGRESS|' then begin
    Tail := Copy(S, 10, Length(S));
    Split := Pos('|', Tail);
    if Split > 0 then begin
      Number := Copy(Tail, 1, Split - 1);
      ProgressPage.SetProgress(StrToIntDef(Number, 0), 100);
      ProgressPage.SetText(CustomMessage('VerifyComponents'), Copy(Tail, Split + 1, Length(Tail)));
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var Code: Integer; Args, PreviousDirectory: String;
begin
  Result := '';
  if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{' + '{#AppIdentity}' + '}_is1',
     'InstallLocation', PreviousDirectory) then begin
    if CompareText(RemoveBackslashUnlessRoot(PreviousDirectory), RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) <> 0 then begin
      Result := FmtMessage(CustomMessage('OtherDirectory'), [PreviousDirectory]);
      exit;
    end;
  end;
  ComponentRoot := ExpandConstant('{param:COMPONENTROOT|{localappdata}\Aester\components}');
  DownloadCache := ExpandConstant('{param:DOWNLOADCACHE|' + ComponentRoot + '\downloads}');
  { auto (default): measure the official hosts and the listed mirrors, download from the fastest;
    /DOWNLOADSOURCE=official uses the official URLs only (for GIMP also GIMP's own mirror opencolo.mm.fcix.net),
    without measuring. Any other value means auto. Files are SHA-256 pinned. }
  if CompareText(ExpandConstant('{param:DOWNLOADSOURCE|auto}'), 'official') = 0 then
    DownloadSource := 'official'
  else
    DownloadSource := 'auto';
  ResultFile := ExpandConstant('{tmp}\aester-prepared.json');
  ProgressPage.Show;
  try
    ProgressPage.SetText(CustomMessage('ExtractPython'), CustomMessage('PrivatePython'));
    ExtractTemporaryFiles('{tmp}\aester-bootstrap\*');
    WorkerParams := '-X utf8 ' + Q(ExpandConstant('{tmp}\aester-bootstrap\installer_worker.py')) +
      ' --component-root ' + Q(ComponentRoot) + ' --app-root ' + Q(ExpandConstant('{app}\versions\{#VersionId}')) +
      ' --cache ' + Q(DownloadCache) + ' --result ' + Q(ResultFile) + ' --download-source ' + DownloadSource;
    if not ExecAndLogOutput(ExpandConstant('{tmp}\aester-bootstrap\python.exe'), WorkerParams + ' --preflight',
      ExpandConstant('{tmp}\aester-bootstrap'), SW_HIDE, ewWaitUntilTerminated, Code, @ProgressLog) or (Code <> 0) then begin
      Result := CustomMessage('InstallBusy');
      exit;
    end;
    Args := WorkerParams;
#ifdef IncludeRetouch
    if WizardIsComponentSelected('gimp') then Args := Args + ' --gimp';
    if WizardIsComponentSelected('retouch_skill') then Args := Args + ' --retouch-skill';
#endif
#ifdef IncludeFinite
    if WizardIsComponentSelected('finite') then begin
      ProgressPage.SetText(CustomMessage('ExtractFinite'), CustomMessage('ExtractFiniteDetail'));
      ExtractTemporaryFiles('{tmp}\aester-finite\*');
      Args := Args + ' --finite-bundle ' + Q(ExpandConstant('{tmp}\aester-finite'));
    end;
#endif
    Code := RunWorker(ExpandConstant('{tmp}\aester-bootstrap\python.exe'), ExpandConstant('{tmp}\aester-bootstrap'), Args, True);
    if Code = AesterWorkerCancelled then
      Result := FmtMessage(CustomMessage('PrepareCancelled'), [DownloadCache])
    else if Code <> 0 then
      Result := FmtMessage(CustomMessage('PrepareFailed'), [ComponentRoot + '\install-failure.json'])
    else
      ShowPreparationNotices;
  finally
    ProgressPage.Hide;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Support, Outcome, Args, Detail, Archive: String;
  Code: Integer;
  RawText: AnsiString;
  ClearCache, ClearSettings: Boolean;
begin
  if CurUninstallStep <> usUninstall then exit;
  { Inno confirmation already happened. Abort here stops Inno's own deletion. }
  Support := ExpandConstant('{tmp}\aester-uninstall');
  Outcome := ExpandConstant('{tmp}\aester-uninstall-options.json');
  try
    CopyUninstallSupport(Support);
    Args := '-B -X utf8 ' + Q(Support + '\uninstall_worker.py') + ' --root ' + Q(ExpandConstant('{app}')) +
      ' --language ' + Q(ActiveLanguage);
    if not Exec(Support + '\python.exe', Args + ' --inspect --result ' + Q(Outcome), Support,
      SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
      RaiseException(CustomMessage('InspectFailed'));
    if ExpandConstant('{param:PREVIEWONLY|0}') = '1' then begin
      Log('Aester uninstall preview completed; cancelled before destructive cleanup.');
      Abort;
    end;
    Detail := CustomMessage('KeepPhotos');
    if LoadStringFromFile(ChangeFileExt(Outcome, '.txt'), RawText) then Detail := UTF8Decode(RawText);
    ClearCache := ExpandConstant('{param:CLEARSAFECACHE|0}') = '1';
    ClearSettings := ExpandConstant('{param:CLEARSETTINGS|0}') = '1';
    if not UninstallSilent then begin
      if not ShowRemovalOptions(Detail, ClearCache, ClearSettings) then Abort;
    end;
    Outcome := ExpandConstant('{app}\uninstall-report.json');
    Args := Args + ' --result ' + Q(Outcome);
    if ClearCache then Args := Args + ' --clear-safe-cache';
    if ClearSettings then Args := Args + ' --clear-settings';
    if not Exec(Support + '\python.exe', Args, Support, SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then begin
      Detail := CustomMessage('UninstallFailed');
      if LoadStringFromFile(ChangeFileExt(Outcome, '.txt'), RawText) then Detail := UTF8Decode(RawText);
      Log(Detail);
      if not UninstallSilent then MsgBox(Detail, mbError, MB_OK);
      Abort;
    end;
    Archive := ExpandConstant('{localappdata}\Aester\UninstallLogs'); ForceDirectories(Archive);
    FileCopy(Outcome, Archive + '\uninstall-' + GetDateTimeString('yyyymmdd-hhnnss', '-', ':') + '.json', False);
  except
    Log('Aester guarded uninstall stopped: ' + GetExceptionMessage);
    if (not UninstallSilent) and (GetExceptionMessage <> '') then MsgBox(GetExceptionMessage, mbError, MB_OK);
    Abort;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Code: Integer;
begin
  if CurStep = ssPostInstall then begin
    ProgressPage.Show;
    try
      ProgressPage.SetText(CustomMessage('Activating'), '');
      Code := RunWorker(ExpandConstant('{tmp}\aester-bootstrap\python.exe'), ExpandConstant('{tmp}\aester-bootstrap'),
        WorkerParams + ' --activate ' + Q(ExpandConstant('{app}\install-state.json')), False);
    finally
      ProgressPage.Hide;
    end;
    if Code <> 0 then
      RaiseException(FmtMessage(CustomMessage('ActivationFailed'), [ComponentRoot + '\install-failure.json']));
    if not SaveStringToFile(ExpandConstant('{app}\install-language.txt'), ActiveLanguage, False) then
      RaiseException(CustomMessage('LanguageWriteFailed'));
  end;
end;
