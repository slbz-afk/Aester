"""Independent regression for stale portable backend preferences.

Real Qt controls, temporary INI preferences, and real MCP config/host admission.
No real registry, actual installation, backend executable, or model is used.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from aester_desktop import gui
from aester_desktop.backend_preferences import apply_backend_defaults
from starstack.mcp_server import WorkspaceTools

NEW_ADAPTER={'channel_order':'bgr','value_range':'-1..1','layout':'nhwc','stitch':'blend'}
OLD_ADAPTER={'channel_order':'rgb','value_range':'0..1','layout':'nchw','stitch':'crop'}
MODEL_CONTROLS={'channel_order':'model_channel_order','value_range':'model_range','layout':'model_layout','stitch':'model_stitch'}


def file(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'Independent synthetic file, never executed or loaded')
    return str(path)


def adapter(window):
    return {key:getattr(window,control).currentData() for key,control in MODEL_CONTROLS.items()}


def snapshot(settings):
    settings.sync()
    return {key:settings.value(key) for key in settings.allKeys()}


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def windows(app,monkeypatch,tmp_path):
    for key in ('STARSTACK_SIRIL','STARSTACK_STAR_MODEL','STARSTACK_STAR_ADAPTER','STARSTACK_STACK_ENGINE',
                'STARSTACK_INPUT_ROOT','STARSTACK_OUTPUT_ROOT','STARSTACK_CACHE_ROOT','AESTER_CACHE_ROOT'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(gui.MainWindow,'check_backend',lambda self:None)
    opened=[]
    def create(values=None,ini=None):
        ini=ini or tmp_path/f'preferences-{len(opened)}.ini'
        preferences=QSettings(str(ini),QSettings.IniFormat)
        if values is not None:
            for key,value in {'output_directory':str(tmp_path/'works'),'storage/cache_root':str(tmp_path/'cache'),**values}.items():
                preferences.setValue(key,value)
            preferences.sync()
        before=snapshot(preferences)
        w=gui.MainWindow(preferences=preferences)
        opened.append(w)
        return w,preferences,ini,before
    yield create
    for w in opened:
        w._color_timer.stop();w.timer.stop();w._close_timer.stop()
        for dialog in w._advanced_dialogs.values():dialog.close()
        w.close()
    app.processEvents()


def environment(monkeypatch,tmp_path,*,model=True):
    siril=file(tmp_path/'new installation 中文/siril-cli.exe')
    monkeypatch.setenv('STARSTACK_SIRIL',siril)
    model_path=file(tmp_path/'new installation 中文/nox.onnx') if model else None
    if model:
        monkeypatch.setenv('STARSTACK_STAR_MODEL',model_path)
        monkeypatch.setenv('STARSTACK_STAR_ADAPTER',json.dumps(NEW_ADAPTER))
    return siril,model_path


def stale_preferences(tmp_path):
    return {'siril_path':str(tmp_path/'deleted portable/siril-cli.exe'),
            'star_model_path':str(tmp_path/'deleted portable/nox.onnx'),
            **{'model/'+k:v for k,v in OLD_ADAPTER.items()}}


def test_actual_startup_migrates_missing_paths_with_complete_adapter_and_restarts(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    values=stale_preferences(tmp_path);values.update({'other/untouched':['alpha','beta'],'appearance/theme':'paper',
        'mcp/input_root':str(tmp_path/'input'),'stack_engine_path':file(tmp_path/'custom/finite.exe')})
    w,p,ini,before=windows(values)
    assert w.siril_path.text()==siril and w.star_model_path.text()==model
    assert adapter(w)==NEW_ADAPTER
    migrated={item['key']:item for item in w.experience.backend_migrations}
    assert set(migrated)=={'siril_path','star_model_path'}
    assert migrated['siril_path']['previous']==values['siril_path']
    assert migrated['star_model_path']['current']==model
    after=snapshot(p)
    expected={**before,'siril_path':siril,'star_model_path':model,**{'model/'+k:v for k,v in NEW_ADAPTER.items()}}
    assert after==expected  # No full preference rewrite or cleared unrelated keys.
    assert '失效' in w.message_label.text()
    w2,p2,_,_=windows(None,ini)
    assert w2.siril_path.text()==siril and w2.star_model_path.text()==model
    assert adapter(w2)==NEW_ADAPTER
    assert w2.experience.backend_migrations==[]
    assert snapshot(p2)==expected


def test_gui_mcp_config_and_real_host_share_migrated_backends(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    engine=file(tmp_path/'external finite/engine.exe')
    w,p,_,_=windows({**stale_preferences(tmp_path),'stack_engine_path':engine})
    source=tmp_path/'RAW';source.mkdir();w.input_dir.setText(str(source))
    w.experience.show_connection()
    config=w.experience._config;assert config is not None
    env=config['mcpServers']['aester']['env']
    assert env['STARSTACK_SIRIL']==siril and env['STARSTACK_STAR_MODEL']==model
    assert env['STARSTACK_STACK_ENGINE']==engine
    assert json.loads(env['STARSTACK_STAR_ADAPTER'])==NEW_ADAPTER
    assert 'deleted portable' not in json.dumps(config)
    host=WorkspaceTools(env['STARSTACK_INPUT_ROOT'],env['STARSTACK_OUTPUT_ROOT'],
        cache_root=env['STARSTACK_CACHE_ROOT'],siril_path=env['STARSTACK_SIRIL'],
        stack_engine_path=env['STARSTACK_STACK_ENGINE'],star_model_path=env['STARSTACK_STAR_MODEL'],
        star_backend_options=json.loads(env['STARSTACK_STAR_ADAPTER']))
    assert host.siril_path==w.siril_path.text()
    assert host.star_model_path==w.star_model_path.text()
    assert host.stack_engine_path==w.stack_engine_path.text()
    assert host.star_backend_options==adapter(w)


def test_existing_external_selection_and_adapter_are_preserved(windows,monkeypatch,tmp_path):
    environment(monkeypatch,tmp_path)
    values={'siril_path':file(tmp_path/'external/siril.exe'),'star_model_path':file(tmp_path/'external/custom.onnx'),
            'stack_engine_path':file(tmp_path/'external/finite.exe'),**{'model/'+k:v for k,v in OLD_ADAPTER.items()}}
    w,p,_,before=windows(values)
    assert w.siril_path.text()==values['siril_path']
    assert w.star_model_path.text()==values['star_model_path']
    assert w.stack_engine_path.text()==values['stack_engine_path']
    assert adapter(w)==OLD_ADAPTER and w.experience.backend_migrations==[]
    assert snapshot(p)==before


@pytest.mark.parametrize('invalid_default',['absent','missing','directory'])
def test_no_valid_replacement_preserves_failure_and_preferences(windows,monkeypatch,tmp_path,invalid_default):
    if invalid_default!='absent':
        path=tmp_path/'not-a-backend'
        if invalid_default=='directory':path.mkdir()
        monkeypatch.setenv('STARSTACK_SIRIL',str(path));monkeypatch.setenv('STARSTACK_STAR_MODEL',str(path))
        monkeypatch.setenv('STARSTACK_STAR_ADAPTER',json.dumps(NEW_ADAPTER))
    values=stale_preferences(tmp_path);w,p,_,before=windows(values)
    assert w.siril_path.text()==values['siril_path'] and w.star_model_path.text()==values['star_model_path']
    assert adapter(w)==OLD_ADAPTER and w.experience.backend_migrations==[]
    assert snapshot(p)==before


@pytest.mark.parametrize('payload',['{','null','[]','{}',json.dumps({'channel_order':'bgr'}),
    *[json.dumps({**NEW_ADAPTER,key:'invalid'}) for key in NEW_ADAPTER]])
def test_invalid_or_incomplete_adapter_never_partially_changes_model(windows,monkeypatch,tmp_path,payload):
    siril,model=environment(monkeypatch,tmp_path);monkeypatch.setenv('STARSTACK_STAR_ADAPTER',payload)
    values=stale_preferences(tmp_path);w,p,_,before=windows(values)
    assert w.siril_path.text()==siril
    assert w.star_model_path.text()==values['star_model_path'] and adapter(w)==OLD_ADAPTER
    after=snapshot(p);assert after=={**before,'siril_path':siril}
    assert {item['key'] for item in w.experience.backend_migrations}=={'siril_path'}


def test_missing_optional_finite_engine_is_not_silently_disabled(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    missing=str(tmp_path/'missing special finite.exe');fallback=file(tmp_path/'installed/other finite.exe')
    monkeypatch.setenv('STARSTACK_STACK_ENGINE',fallback)
    w,p,_,_=windows({**stale_preferences(tmp_path),'stack_engine_path':missing})
    assert w.siril_path.text()==siril and w.star_model_path.text()==model
    assert w.stack_engine_path.text()==missing and p.value('stack_engine_path')==missing
    source=tmp_path/'RAW';source.mkdir();w.input_dir.setText(str(source));w.experience.show_connection()
    assert w.experience._config is None  # The unsupported choice stays visible for the user to fix.


def test_new_interactive_invalid_choice_is_not_overridden_during_config_refresh(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    w,p,_,_=windows(stale_preferences(tmp_path))
    source=tmp_path/'RAW';source.mkdir();w.input_dir.setText(str(source));w.experience.show_connection()
    assert w.experience._config is not None
    chosen=str(tmp_path/'user just chose absent.exe');w.siril_path.setText(chosen)
    w.experience._refresh_connection()
    assert w.siril_path.text()==chosen and w.experience._config is None
    assert p.value('siril_path')==siril  # Startup migration is not repeated during interactive editing.


def test_one_valid_external_backend_does_not_block_other_missing_backend_recovery(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    external=file(tmp_path/'external/model.onnx')
    w,p,_,_=windows({**stale_preferences(tmp_path),'star_model_path':external})
    assert w.siril_path.text()==siril and w.star_model_path.text()==external and adapter(w)==OLD_ADAPTER


def test_blank_fields_adopt_defaults_without_enabling_effects(windows,monkeypatch,tmp_path):
    siril,model=environment(monkeypatch,tmp_path)
    w,p,_,_=windows({})
    assert w.siril_path.text()==siril and w.star_model_path.text()==model and adapter(w)==NEW_ADAPTER
    assert not w.stars_enabled.isChecked() and not w.trails_enabled.isChecked() and not w.milkyway_enabled.isChecked()


def test_explicit_environment_mapping_and_no_preferences_supported(windows,tmp_path):
    w,p,_,before=windows({})
    w._preferences=None
    siril=file(tmp_path/'provided/siril.exe');model=file(tmp_path/'provided/model.onnx')
    try:
        changes=apply_backend_defaults(w,{'STARSTACK_SIRIL':siril,'STARSTACK_STAR_MODEL':model,
            'STARSTACK_STAR_ADAPTER':json.dumps(NEW_ADAPTER)})
        assert {x['key'] for x in changes}=={'siril_path','star_model_path'}
        assert w.siril_path.text()==siril and w.star_model_path.text()==model and adapter(w)==NEW_ADAPTER
        assert snapshot(p)==before
    finally:w._preferences=p
