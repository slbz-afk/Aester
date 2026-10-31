"""Codex independent contract tests for the unmodified DeepSeek pure function."""
import copy
import math
from fractions import Fraction
import pytest
from starstack.exposure_summary import summarize_exposures


def test_unknown_values_do_not_hide_known_selected_total():
    exposures = [20, None, 10.5, 1e-8]
    chosen = [2, 0]
    before = copy.deepcopy((exposures, chosen))
    assert summarize_exposures(exposures, chosen) == {
        'input_count':4, 'included_count':2, 'input_total_seconds':None,
        'included_total_seconds':30.5, 'unknown_input_count':1, 'unknown_included_count':0}
    assert (exposures, chosen) == before
    assert summarize_exposures(exposures, [1])['included_total_seconds'] is None


@pytest.mark.parametrize('bad', [True, False, 0, -1, -0.0, float('nan'),float('inf'),-float('inf'), '20', {}, [], 10**1000])
def test_unselected_invalid_exposure_rejected(bad):
    with pytest.raises(ValueError): summarize_exposures([20,bad], [0])


@pytest.mark.parametrize('indices', [[True], [0.0], ['0'], [-1], [2], [0,0], (0,), None])
def test_membership_must_be_explicit_valid_unique_indices(indices):
    with pytest.raises(ValueError): summarize_exposures([20,30], indices)


def test_types_empty_overflow_and_accurate_sum():
    class Number(int): pass
    class Array(list): pass
    for values, chosen in [(Array([1]),[0]),([Number(1)],[0]),([1],[Number(0)]),([1],Array([0]))]:
        with pytest.raises(ValueError): summarize_exposures(values,chosen)
    assert summarize_exposures([],[]) == {'input_count':0,'included_count':0,'input_total_seconds':0.0,
                                         'included_total_seconds':0.0,'unknown_input_count':0,'unknown_included_count':0}
    tiny=math.ulp(0.0)
    values=[tiny,1e16,1,1,1,1e-10]
    result=summarize_exposures(values,list(range(len(values))))
    expected=float(sum((Fraction(v) for v in values),Fraction()))
    assert result['input_total_seconds']==expected==result['included_total_seconds']
    assert summarize_exposures([tiny],[0])['included_total_seconds']==tiny
    with pytest.raises(ValueError): summarize_exposures([1e308,1e308],[])
    with pytest.raises(ValueError): summarize_exposures([None,1e308,1e308],[1,2])
