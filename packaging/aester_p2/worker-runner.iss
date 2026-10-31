{ Runs the Aester installer worker without blocking Setup's window.

  Script Exec/ExecAndLogOutput disable every Setup window while they wait, so
  the wizard could not be moved, minimized or closed for minutes (user report,
  2026-09-27). Here the worker starts with ewNoWait and a loop polls its files:
  progress (percent, headline, detail), status (state, exit code) and pid.
  Cancel/close asks first, then requests a stop at the worker's next safe point.

  Requires before inclusion: ProgressPage (TOutputProgressWizardPage), Q() and
  WinCloseHandle. Used by Aester.iss and by the window-behaviour test Setup. }

const
  AesterSynchronize = $00100000;
  AesterQueryLimited = $1000;
  AesterWaitObject0 = 0;
  AesterScClose = $F060;
  AesterMfEnabled = 0;
  AesterMfGrayed = 1;
  AesterWorkerCancelled = 3;
  AesterWorkerLost = -2;

var
  WorkerProgressFile, WorkerStatusFile, WorkerPidFile, WorkerCancelFile: String;
  CancelAllowed, CancelRequested, CancelSignalled: Boolean;

function WinOpenProcess(Access: Cardinal; Inherit: Boolean; ProcessId: Cardinal): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WinWaitForSingleObject(Handle: THandle; Milliseconds: Cardinal): Cardinal;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function WinGetExitCodeProcess(Process: THandle; var ExitCode: Cardinal): Boolean;
  external 'GetExitCodeProcess@kernel32.dll stdcall';
function WinGetSystemMenu(Wnd: HWND; Revert: Boolean): THandle;
  external 'GetSystemMenu@user32.dll stdcall';
function WinEnableMenuItem(Menu: THandle; Item, Flags: Cardinal): Integer;
  external 'EnableMenuItem@user32.dll stdcall';
function WinGetTickCount: Cardinal;
  external 'GetTickCount@kernel32.dll stdcall';

procedure SetCancelAvailable(const Enable: Boolean);
var Flags: Cardinal;
begin
  CancelAllowed := Enable;
  WizardForm.CancelButton.Visible := Enable;
  WizardForm.CancelButton.Enabled := Enable;
  if Enable then Flags := AesterMfEnabled else Flags := AesterMfGrayed;
  WinEnableMenuItem(WinGetSystemMenu(WizardForm.Handle, False), AesterScClose, Flags);
end;

procedure ShowWorkerProgress;
var Lines: TArrayOfString; Percent: Integer;
begin
  { LoadStringsFromFile decodes UTF-8 and splits lines. (Pos(#10, S) on the
    decoded text counted Chinese characters in ANSI bytes and cut the lines.) }
  if LoadStringsFromFile(WorkerProgressFile, Lines) and (GetArrayLength(Lines) >= 3) then begin
    Percent := StrToIntDef(Trim(Lines[0]), -1);
    if (Percent >= 0) and (Percent <= 100) then begin
      { Set the labels directly: SetText shortens its second line as a file
        path, which cut "PySide6 (54 / 120 MB)" at the slash. }
      ProgressPage.Msg1Label.Caption := Lines[1];
      ProgressPage.Msg2Label.Caption := Lines[2];
      ProgressPage.SetProgress(Percent, 100);
      exit;
    end;
  end;
  { No new numbers yet: still let the window handle input and repaint. }
  ProgressPage.SetProgress(ProgressPage.ProgressBar.Position, 100);
end;

function ReadWorkerStatus(var Code: Integer): Boolean;
var Lines: TArrayOfString;
begin
  Result := False;
  if not LoadStringsFromFile(WorkerStatusFile, Lines) or (GetArrayLength(Lines) < 2) then exit;
  Code := StrToIntDef(Trim(Lines[1]), AesterWorkerLost);
  Log('Aester worker finished: ' + Trim(Lines[0]) + ' (' + IntToStr(Code) + ')');
  Result := True;
end;

function RunWorker(const Python, WorkDir, Params: String; const AllowCancel: Boolean): Integer;
var
  Started: Cardinal; ResultCode: Integer; Process: THandle; ExitCode: Cardinal; Raw: AnsiString;
begin
  Result := AesterWorkerLost;
  WorkerProgressFile := ExpandConstant('{tmp}\aester-progress.txt');
  WorkerStatusFile := ExpandConstant('{tmp}\aester-status.txt');
  WorkerPidFile := ExpandConstant('{tmp}\aester-worker.pid');
  WorkerCancelFile := ExpandConstant('{tmp}\aester-cancel.request');
  DeleteFile(WorkerProgressFile); DeleteFile(WorkerStatusFile);
  DeleteFile(WorkerPidFile); DeleteFile(WorkerCancelFile);
  CancelRequested := False;
  CancelSignalled := False;
  if not Exec(Python, Params + ' --language ' + Q(ActiveLanguage) +
       ' --progress-file ' + Q(WorkerProgressFile) + ' --status-file ' + Q(WorkerStatusFile) +
       ' --pid-file ' + Q(WorkerPidFile) + ' --cancel-file ' + Q(WorkerCancelFile),
       WorkDir, SW_HIDE, ewNoWait, ResultCode) then begin
    Log('Aester worker could not start: ' + SysErrorMessage(ResultCode));
    exit;
  end;
  SetCancelAvailable(AllowCancel);
  Process := 0;
  Started := WinGetTickCount;
  try
    while True do begin
      ShowWorkerProgress;
      if CancelRequested and not CancelSignalled then begin
        SaveStringToFile(WorkerCancelFile, 'cancel', False);
        CancelSignalled := True;
        Log('Cancellation requested by the user');
      end;
      if ReadWorkerStatus(Result) then break;
      if Process = 0 then begin
        if LoadStringFromFile(WorkerPidFile, Raw) then begin
          Process := WinOpenProcess(AesterSynchronize or AesterQueryLimited, False,
            StrToIntDef(Trim(String(Raw)), 0));
          if Process = 0 then begin
            { It already ended; its status file tells how. }
            Sleep(300);
            if not ReadWorkerStatus(Result) then Result := AesterWorkerLost;
            break;
          end;
        end else if WinGetTickCount - Started > 120000 then begin
          Log('Aester worker did not report its process id');
          break;
        end;
      end else if WinWaitForSingleObject(Process, 0) = AesterWaitObject0 then begin
        if not ReadWorkerStatus(Result) then begin
          if WinGetExitCodeProcess(Process, ExitCode) then
            Log('Aester worker ended without a status, exit ' + IntToStr(ExitCode));
          Result := AesterWorkerLost;
        end;
        break;
      end;
      Sleep(100);
    end;
  finally
    if Process <> 0 then WinCloseHandle(Process);
    SetCancelAvailable(False);
  end;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if (ProgressPage <> nil) and (CurPageID = ProgressPage.ID) then begin
    { Never end Setup under a running worker; ask it to stop at a safe point. }
    Cancel := False;
    if CancelAllowed and not CancelRequested then
      if MsgBox(CustomMessage('CancelPrepareConfirm'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        CancelRequested := True;
  end;
end;

