from pathlib import Path
import copy,json,sys
from types import SimpleNamespace
import numpy as np
import pytest
from starstack import cli,core,transient_stage as ts,transient_review as review,transient_trails as tt
from starstack.mcp_server import WorkspaceTools


def candidate(cid,decision='omit',status='measured',label=None,override=None,faded=False):
    return {'id':cid,'stem':'light_00001','source':'auto','decision':decision,'label':label,'override':override,
        'measurement':{'status':status,'ends':{},'quality':{}},'suggestion':{'class':'unknown'},
        'evidence':{'light_curve':{'ends':{'stop':'faded'} if faded else {}}},'geometry':{},'provenance':{}}


def record():
    return {'candidates':[candidate('z',faded=True),candidate('u',status='too_short'),candidate('a'),
        candidate('retained','keep'),candidate('manual',override='omit')],
        'summary':{'candidates':9999,'recall_review':{'message':'stale 9999','candidate_ids':[]}},
        'policy':{'unknown':'omit'},'frames_scanned':1}


def args(tmp_path,**options):
    return SimpleNamespace(job=str(tmp_path/'job'),label=None,override=None,policy=None,mark=None,note=None,
        full=options.get('full',False),needs_review=options.get('needs_review',False))


def test_summary_dynamic_and_record_unchanged():
    r=record();before=copy.deepcopy(r);s=ts.summary(r)
    assert s['candidates']==5 and s['recall_review']['automatic_omitted']=={'total':3,'measured_unknown':2,'unmeasured':1,'high_priority':1}
    assert s['recall_review']['candidate_ids']==['z','a','u']
    assert r==before


@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('filtered',[False,True])
def test_cli_old_record_dynamic_without_mutation(tmp_path,monkeypatch,full,filtered):
    r=record();before=copy.deepcopy(r);job={'job_dir':str(tmp_path/'job'),'artifacts':[]}
    monkeypatch.setattr(cli,'load_job',lambda p:job)
    monkeypatch.setattr(cli,'recompose_transients',lambda *a,**k:pytest.fail('listing must not recompose'))
    monkeypatch.setattr(ts,'read_record',lambda *a,**k:r)
    got=cli._transients(args(tmp_path,full=full,needs_review=filtered))
    s=got['transients']['summary'] if full else got['summary']
    assert s['candidates']==5 and s['recall_review']['automatic_omitted']['unmeasured']==1
    rows=got['transients']['candidates'] if full else got['candidates']+got['not_measured']
    assert len(rows)==(3 if filtered else 5)
    assert {x['id'] for x in rows}==({'z','a','u'} if filtered else {'z','a','u','retained','manual'})
    assert all('recall_review' in x for x in rows)
    if filtered:
        order=[x['id'] for x in rows] if full else got['review_order']
        assert order==['z','a','u']
    assert r==before and got['summary_scope']=='all_candidates'


def test_cli_parser_needs_review(monkeypatch,capsys):
    calls=[];monkeypatch.setattr(sys,'argv',['starstack','transients','job','--needs-review'])
    monkeypatch.setattr(cli,'_transients',lambda a:calls.append(a) or {'listed':True})
    assert cli.main()==0 and calls[0].needs_review is True
    assert json.loads(capsys.readouterr().out)['listed'] is True


def test_compact_keeps_unfulfilled_request_visible():
    c=candidate('request',status='no_signal');c['requested_decision']='keep'
    r=cli.candidate_view(c)
    assert r['recall_review']['status']=='requested_keep_unmeasured' and r['recall_review']['priority']=='high'
    assert r['decision']=='omit'


@pytest.mark.parametrize('filtered',[False,True])
def test_mcp_refreshes_summary_and_orders_without_mutation(tmp_path,monkeypatch,filtered):
    r=record();before=copy.deepcopy(r);service=WorkspaceTools(input_root=tmp_path/'inputs',output_root=tmp_path/'outputs')
    monkeypatch.setattr(service,'_job',lambda p:(tmp_path/'outputs/job',{}))
    monkeypatch.setattr(service,'_artifacts',lambda *a:[])
    monkeypatch.setattr(ts,'read_record',lambda *a,**k:r)
    got=service.list_transients('job',needs_review=filtered)
    assert got['summary']['candidates']==5 and got['summary']['recall_review']['automatic_omitted']['unmeasured']==1
    if filtered:assert [x['id'] for x in got['candidates']]==['z','a','u']
    assert len(got['candidates'])==(3 if filtered else 5) and r==before
    with pytest.raises(ValueError):service.list_transients('job',needs_review=1)


def test_core_warning_includes_unmeasured_and_no_recall_certification():
    r={'candidates':[candidate('u',status='too_short')]}
    msg=core._transients_message({'summary':ts.summary(r)})
    assert '未能测量' in msg and '自动省略' in msg and '不能保证没有遗漏' in msg


def test_legacy_summary_does_not_silently_claim_audit():
    msg=core._transients_message({'summary':{'candidates':4,'measured':0,'kept':[]}})
    assert '旧摘要' in msg and '未测量' in msg and '不能保证没有遗漏' in msg


@pytest.mark.parametrize('field,value', [('label',[]),('effective_label',[]),('suggestion',[]),('suggestion',{'class':[]})])
def test_malformed_class_fields_clear_error(field,value):
    c=candidate('invalid');c[field]=value
    with pytest.raises(ValueError):review.review_candidate(c)


@pytest.mark.parametrize('missing',['id','decision'])
def test_bad_record_not_silently_fabricated(missing):
    c=candidate('bad');c.pop(missing)
    with pytest.raises(ValueError):ts.summary({'candidates':[c]})


def test_recompose_replaces_old_message(tmp_path,monkeypatch):
    from astropy.io import fits
    from starstack import imaging,registration_calibration,stack_range,transient_revision
    directory=tmp_path/'job';work=directory/'work';work.mkdir(parents=True);(directory/'products').mkdir()
    fits.PrimaryHDU(np.zeros((3,8,8),np.float32)).writeto(work/'sky_master.fit')
    rec=record();job={'state':'succeeded','job_dir':str(directory),'mask':None,'inputs':[],
                    'message':'stale 9999 omissions','artifacts':[]}
    diag={'summary':ts.summary(rec)};captured={}
    monkeypatch.setattr(stack_range,'verify_receipt',lambda *a:None)
    monkeypatch.setattr(ts,'read_record',lambda *a,**k:copy.deepcopy(rec))
    monkeypatch.setattr(ts,'update_record',lambda work,r,**k:r)
    monkeypatch.setattr(registration_calibration,'load_calibration',lambda *a:None)
    monkeypatch.setattr(imaging,'read_rgb',lambda *a:np.zeros((8,8,3),np.float32))
    monkeypatch.setattr(core,'_export_transients',lambda *a,**k:(copy.deepcopy(diag),[]))
    monkeypatch.setattr(transient_revision,'prepare',lambda *a:{'record':'newrecord','id':'newrevision','sha256':'hash'})
    monkeypatch.setattr(core,'_update',lambda d,**k:captured.update(k))
    monkeypatch.setattr(core,'load_job',lambda *a:captured)
    result=core._recompose_locked(directory,job,None,None,None,None,None)
    assert 'stale' not in result['message'] and '9999' not in result['message']
    assert '3 条候选被自动省略' in result['message'] and '1 条未能测量' in result['message']


def test_classification_conditions_unchanged_on_original_records():
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/starstack').is_dir())
    # Keep the actual measured classification inputs, without keeping the retired
    # 144-frame pixel cache. A missing checked-in fixture must fail, not skip.
    path=root/'docs/testdata/HISTORICAL_TRANSIENT_CLASSIFICATION_20260925.json'
    r=json.loads(path.read_text(encoding='utf-8'))
    assert len(r['candidates'])==r['source_candidates']==1206
    for c in r['candidates']:assert tt.suggest(c)['class']==c['suggestion']['class']
    marked=next(c for c in r['candidates'] if c['id']=='light_00009-8')
    reasons=' '.join(tt.suggest(marked)['reasons'])
    assert 'absence is not established' in reasons and 'does not exclude a meteor' in reasons
