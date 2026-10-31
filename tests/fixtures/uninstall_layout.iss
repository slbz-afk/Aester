; Harmless native harness: no application files, registration or cleanup.
; Includes the exact production form; the test hook only records its layout.
#ifndef UiSource
  #error UiSource is required
#endif
#ifndef FontSize
  #define FontSize "9"
#endif
#define AesterUninstallUiHarness
[Setup]
AppName=Aester uninstall UI test
AppVersion=1
AppId=Aester-Uninstall-UI-Harness
CreateAppDir=no
DefaultDirName={tmp}\aester-ui-harness-unused
Uninstallable=no
PrivilegesRequired=lowest
WizardStyle=modern dynamic
WizardSizePercent=115
DisableReadyPage=yes
DisableFinishedPage=yes
OutputDir={#BuildOutput}
OutputBaseFilename=uninstall-ui-font-{#FontSize}
[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "zh_CN"; MessagesFile: "{#UiSource}\ChineseSimplified.isl"
[LangOptions]
DialogFontSize={#FontSize}
#include AddBackslash(UiSource) + "installer-messages.iss"
[Code]
var Report: String;

procedure RecordControl(const Name, Caption: String; Control: TControl; Font: TFont; const IndicatorWidth: Integer);
var Bitmap: TBitmap; TextWidth, TextHeight: Integer; ControlHandle: HWND;
begin
  ControlHandle := TWinControl(Control).Handle;
  Bitmap := TBitmap.Create;
  try
    Bitmap.Canvas.Font.Assign(Font);
    TextWidth := Bitmap.Canvas.TextWidth(Caption);
    TextHeight := Bitmap.Canvas.TextHeight(Caption);
    Report := Report + Name + '=' + IntToStr(Control.Left) + ',' + IntToStr(Control.Top) + ',' +
      IntToStr(Control.Width) + ',' + IntToStr(Control.Height) + ',' +
      IntToStr(TextWidth) + ',' + IntToStr(TextHeight) + ',' + IntToStr(IndicatorWidth) + #13#10;
  finally
    Bitmap.Free;
  end;
end;

procedure RecordUninstallLayout(Form: TSetupForm; Memo: TNewMemo;
  CacheChoice: TNewCheckBox; CacheNote: TNewStaticText;
  SettingsChoice: TNewCheckBox; SettingsNote: TNewStaticText;
  ContinueButton, CancelButton: TNewButton);
var LegacyCheck: TNewCheckBox; LegacyButton: TNewButton;
begin
  Report := 'language=' + ActiveLanguage + #13#10 + 'font_size=' + IntToStr(Form.Font.Size) + #13#10 +
    'font_height=' + IntToStr(Form.Font.Height) + #13#10 + 'font_ppi=' + IntToStr(Form.Font.PixelsPerInch) + #13#10 +
    'form=' + IntToStr(Form.ClientWidth) + ',' + IntToStr(Form.ClientHeight) + #13#10;
  RecordControl('memo', '', Memo, Form.Font, 0);
  RecordControl('cache', CacheChoice.Caption, CacheChoice, CacheChoice.Font, ScaleX(24));
  RecordControl('cache_note', '', CacheNote, CacheNote.Font, 0);
  RecordControl('settings', SettingsChoice.Caption, SettingsChoice, SettingsChoice.Font, ScaleX(24));
  RecordControl('settings_note', '', SettingsNote, SettingsNote.Font, 0);
  RecordControl('continue', ContinueButton.Caption, ContinueButton, ContinueButton.Font, ScaleX(12));
  RecordControl('cancel', CancelButton.Caption, CancelButton, CancelButton.Font, ScaleX(12));
  Report := Report + 'cache_default=' + IntToStr(Ord(CacheChoice.Checked)) + #13#10 +
    'settings_default=' + IntToStr(Ord(SettingsChoice.Checked)) + #13#10;
  LegacyCheck := TNewCheckBox.Create(Form); LegacyCheck.Parent := Form;
  LegacyButton := TNewButton.Create(Form); LegacyButton.Parent := Form;
  RecordControl('legacy_check', CacheChoice.Caption, LegacyCheck, LegacyCheck.Font, 0);
  RecordControl('legacy_button', ContinueButton.Caption, LegacyButton, LegacyButton.Font, 0);
  LegacyCheck.Free; LegacyButton.Free;
  SaveStringToFile(ExpandConstant('{param:REPORT}'), UTF8Encode(Report), False);
end;

#include AddBackslash(UiSource) + "uninstall-ui.iss"
procedure InitializeWizard;
var ClearCache, ClearSettings: Boolean;
begin
  ClearCache := False; ClearSettings := False;
  ShowRemovalOptions(CustomMessage('KeepPhotos') + #13#10 + #13#10 +
    '1.29 GB / 14,019 files' + #13#10 + CustomMessage('CacheNote'), ClearCache, ClearSettings);
end;
