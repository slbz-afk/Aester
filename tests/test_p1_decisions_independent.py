from copy import deepcopy
import pytest
from starstack.review_decisions import normalize_decisions as norm


class SDict(dict): pass
class SList(list): pass
class SStr(str): pass


def test_intent_is_detached_and_unknown_is_not_omit():
    source = {' b ': {'label': 'unknown', 'override': 'keep', 'note': '\n银河\t'},
              'a': {'label': None}, 'c': {'override': None}, 'd': {'note': ''}}
    before = deepcopy(source)
    out = norm(source, ['a', ' b ', 'c', 'd'])
    assert out == before and list(out) == [' b ', 'a', 'c', 'd']
    out[' b ']['label'] = 'satellite'
    assert source == before
    assert norm({}, []) == {}


@pytest.mark.parametrize('ids', [None, (), True, SList(['x']), ['x','x'], [''], [' '],
    ['a\0'], ['a\x7f'], [True], [1], [SStr('x')], ['x'*257], ['a']*5001])
def test_bad_candidate_ids(ids):
    with pytest.raises(ValueError): norm({}, ids)


@pytest.mark.parametrize('decisions', [None, [], True, SDict(), {'y': {'override':'keep'}},
    {SStr('x'): {'label':'unknown'}}, {'x':{}}, {'x':SDict(label='unknown')},
    {'x': {'label': True}}, {'x': {'label': []}}, {'x': {'label':'Unknown'}},
    {'x': {'label':SStr('unknown')}}, {'x': {'override':'auto'}}, {'x': {'override':False}},
    {'x': {'note':None}}, {'x': {'note':SStr('hello')}}, {'x': {'note':'\ud800'}},
    {'x': {'note':'a'*2001}}, {'x': {'extra': 1}}, {'x': {SStr('note'):''}},
    {'x': {'label':'unknown','override':'keep','note':'','extra':None}}])
def test_bad_decisions(decisions):
    with pytest.raises(ValueError): norm(decisions, ['x'])


def test_bounds_are_inclusive():
    ids = [str(i) for i in range(5000)]
    assert len(norm({i:{'label':None} for i in ids}, ids)) == 5000
    assert norm({'x'*256:{'note':'a'*2000}}, ['x'*256])['x'*256]['note'] == 'a'*2000
