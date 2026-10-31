{ Shared by the production uninstaller and the non-destructive UI harness.
  Form construction and every control height must be scaled together: controls
  created after the form do not inherit the DPI-scaled constructor height. }
function ShowRemovalOptions(const Detail: String; var ClearCache, ClearSettings: Boolean): Boolean;
var
  Form: TSetupForm;
  Memo: TNewMemo;
  CacheChoice, SettingsChoice: TNewCheckBox;
  CacheNote, SettingsNote: TNewStaticText;
  ContinueButton, CancelButton: TNewButton;
  Margin, Gap, ButtonWidth, ButtonTop: Integer;
begin
  { The pinned Inno runtime scales constructor dimensions in InitializeFont.
    Supplying ScaleX/ScaleY here double-scales the form on a high-DPI display.
    Native harness coverage checks this rather than relying on the help example. }
  Form := CreateCustomForm(620, 390, True, True);
  try
    Form.Caption := CustomMessage('UninstallTitle');
    Margin := ScaleX(20); Gap := ScaleY(12);
    ContinueButton := TNewButton.Create(Form); ContinueButton.Parent := Form;
    ContinueButton.Caption := CustomMessage('BeginUninstall');
    ContinueButton.Height := ScaleY(32); ContinueButton.ModalResult := mrOk;
    ContinueButton.Default := True;
    CancelButton := TNewButton.Create(Form); CancelButton.Parent := Form;
    CancelButton.Caption := CustomMessage('CancelUninstall');
    CancelButton.Height := ScaleY(32); CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;
    ButtonWidth := Form.CalculateButtonWidth([ContinueButton.Caption, CancelButton.Caption]);
    if ButtonWidth < ScaleX(108) then ButtonWidth := ScaleX(108);
    ContinueButton.Width := ButtonWidth; CancelButton.Width := ButtonWidth;
    ButtonTop := Form.ClientHeight - ScaleY(20) - ContinueButton.Height;
    ContinueButton.Top := ButtonTop; CancelButton.Top := ButtonTop;
    CancelButton.Left := Form.ClientWidth - Margin - ButtonWidth;
    ContinueButton.Left := CancelButton.Left - ScaleX(12) - ButtonWidth;
    ContinueButton.Anchors := [akRight, akBottom]; CancelButton.Anchors := [akRight, akBottom];

    SettingsNote := TNewStaticText.Create(Form); SettingsNote.Parent := Form;
    SettingsNote.Left := Margin + ScaleX(24);
    SettingsNote.Width := Form.ClientWidth - Margin - SettingsNote.Left;
    SettingsNote.AutoSize := False; SettingsNote.WordWrap := True;
    SettingsNote.Caption := CustomMessage('SettingsNote'); SettingsNote.AdjustHeight;
    SettingsNote.Top := ButtonTop - ScaleY(20) - SettingsNote.Height;
    SettingsChoice := TNewCheckBox.Create(Form); SettingsChoice.Parent := Form;
    SettingsChoice.Left := Margin; SettingsChoice.Width := Form.ClientWidth - 2 * Margin;
    SettingsChoice.Height := ScaleY(26); SettingsChoice.Checked := False;
    SettingsChoice.Caption := CustomMessage('ClearSettings');
    SettingsChoice.Top := SettingsNote.Top - SettingsChoice.Height;

    CacheNote := TNewStaticText.Create(Form); CacheNote.Parent := Form;
    CacheNote.Left := SettingsNote.Left; CacheNote.Width := SettingsNote.Width;
    CacheNote.AutoSize := False; CacheNote.WordWrap := True;
    CacheNote.Caption := CustomMessage('CacheNote'); CacheNote.AdjustHeight;
    CacheNote.Top := SettingsChoice.Top - Gap - CacheNote.Height;
    CacheChoice := TNewCheckBox.Create(Form); CacheChoice.Parent := Form;
    CacheChoice.Left := Margin; CacheChoice.Width := SettingsChoice.Width;
    CacheChoice.Height := ScaleY(26); CacheChoice.Checked := False;
    CacheChoice.Caption := CustomMessage('ClearCache');
    CacheChoice.Top := CacheNote.Top - CacheChoice.Height;

    Memo := TNewMemo.Create(Form); Memo.Parent := Form;
    Memo.Left := Margin; Memo.Top := ScaleY(20);
    Memo.Width := Form.ClientWidth - 2 * Margin;
    Memo.Height := CacheChoice.Top - Gap - Memo.Top;
    Memo.ReadOnly := True; Memo.ScrollBars := ssVertical; Memo.Text := Detail;
    Memo.TabStop := False;
    Memo.Anchors := [akLeft, akTop, akRight, akBottom];
    CacheChoice.Anchors := [akLeft, akRight, akBottom]; CacheNote.Anchors := [akLeft, akRight, akBottom];
    SettingsChoice.Anchors := [akLeft, akRight, akBottom]; SettingsNote.Anchors := [akLeft, akRight, akBottom];
    CacheChoice.TabOrder := 0; SettingsChoice.TabOrder := 1;
    ContinueButton.TabOrder := 2; CancelButton.TabOrder := 3;
    Form.ActiveControl := CancelButton;
#ifdef AesterUninstallUiHarness
    { Test builds only: inspect the exact production form without showing it,
      opening an installation or running any cleanup. }
    RecordUninstallLayout(Form, Memo, CacheChoice, CacheNote, SettingsChoice, SettingsNote,
      ContinueButton, CancelButton);
    Result := False;
#else
    Result := Form.ShowModal = mrOk;
#endif
    if Result then begin
      ClearCache := CacheChoice.Checked; ClearSettings := SettingsChoice.Checked;
    end;
  finally
    Form.Free;
  end;
end;
