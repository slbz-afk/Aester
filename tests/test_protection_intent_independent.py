"""Independent protection intent matrix; eligibility is not publication."""
from copy import deepcopy
import pytest

from starstack.protection_intent import resolve_candidate_intents


@pytest.mark.parametrize('label', ['meteor','satellite','aircraft','artifact','unknown'])
@pytest.mark.parametrize('override', [None,'keep','omit'])
@pytest.mark.parametrize('protect', [False,True])
@pytest.mark.parametrize('status', ['measured','no_signal','too_short','ambiguous_target'])
def test_priority_and_failure_preserve_request(label, override, protect, status):
    c={'id':'one','label':label,'suggestion':{'class':'meteor'},'override':override,
       'mark':{'protect':protect},'measurement':{'status':status}}
    original=deepcopy(c)
    result=resolve_candidate_intents([c]); row=result['candidates'][0]
    expected={'keep':'keep','omit':'omit'}.get(override)
    if expected is None: expected='keep' if protect or label=='meteor' else 'omit'
    assert row['requested_decision']==expected
    assert row['decision']==(expected if status=='measured' else 'omit')
    assert row['effective_label']==label
    assert row['protection_status']==('not_requested' if expected=='omit' else
                                     'eligible' if status=='measured' else 'needs_review')
    assert row['label']==label and row['suggestion']=={'class':'meteor'}
    assert c==original
    assert resolve_candidate_intents(result['candidates'],result['policy'])==result
    assert not any(k in row for k in ('fulfilled','applied','success'))


def test_summary_counts_keep_algorithm_and_explicit_intent_separately():
    rows=[{'measurement':{'status':'measured'},'suggestion':{'class':'meteor'}},
          {'measurement':{'status':'no_signal'},'override':'keep'},
          {'measurement':{'status':'measured'},'mark':{'protect':True}},
          {'measurement':{'status':'no_signal'},'mark':{'protect':True},'override':'omit'}]
    r=resolve_candidate_intents(rows)
    assert r['summary']==dict(requested_keep=3,explicit_requested_keep=2,eligible_keep=2,needs_review=1,not_requested=1)
    assert r['candidates'][1]['requested_reason']=='override'
    assert r['candidates'][1]['protection_reason']=='not_measured:no_signal'
    assert r['candidates'][2]['effective_label']=='unknown'
    assert r['candidates'][3]['requested_reason']=='override'


def test_deep_copy_and_stale_derived_values():
    a={'measurement':{'status':'measured','series':[1,2]},'mark':{'protect':True},
       'decision':'omit','protection_status':'fulfilled'}
    r=resolve_candidate_intents([a])
    r['candidates'][0]['measurement']['series'].append(3)
    r['candidates'][0]['mark']['protect']=False
    assert a['measurement']['series']==[1,2] and a['mark']['protect'] is True
    assert r['candidates'][0]['protection_status']=='eligible'


@pytest.mark.parametrize('patch', [{'mark':{'protect':1}},{'mark':{'protect':'true'}},{'mark':[]},
                                  {'measurement':[]},{'measurement':{'status':None}},
                                  {'label':[]},{'label':'comet'},{'override':True},
                                  {'suggestion':[]},{'suggestion':{'class':{}}}])
def test_bad_candidate_is_rejected_without_mutation(patch):
    c={'measurement':{'status':'measured'},**patch}; old=deepcopy(c)
    with pytest.raises(ValueError): resolve_candidate_intents([c])
    assert c==old


@pytest.mark.parametrize('policy',[[],True,{'other':'keep'},{'meteor':True},{'unknown':'maybe'}])
def test_bad_policy(policy):
    with pytest.raises(ValueError): resolve_candidate_intents([],policy)


def test_unknown_policy_and_no_status():
    r=resolve_candidate_intents([{'measurement':{}}],{'unknown':'keep'})['candidates'][0]
    assert r['requested_decision']=='keep' and r['decision']=='omit'
    assert r['protection_reason']=='not_measured:None'
