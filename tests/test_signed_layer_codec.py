"""Root-written independent Python-integer oracles for Worker delivery."""
import numpy as np
import pytest
from starstack.signed_layer_codec import split_signed_codes,compose_layer_codes


@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('reverse',[False,True])
def test_integer_truth_and_independence(endian,reverse):
    values=np.array([-65535,-40001,-1,0,1,60000,65535,-321,403],dtype=endian+'i4').reshape(1,3,3)
    if reverse:values=values[:,::-1]
    original=values.copy();values.flags.writeable=False
    positive,negative=split_signed_codes(values)
    for actual,p,n in zip(values.flat,positive.flat,negative.flat):
        assert int(p)==max(int(actual),0) and int(n)==max(-int(actual),0)
    assert positive.dtype==negative.dtype==np.uint16
    assert positive.flags.c_contiguous and negative.flags.c_contiguous
    positive[:]=0;negative[:]=0
    np.testing.assert_array_equal(values,original)


def test_random_three_images_have_exact_layer_reconstruction():
    rng=np.random.default_rng(581)
    for _ in range(31):
        s,b,d=[rng.integers(0,65536,(7,13,3),dtype=np.uint16) for __ in range(3)]
        r=np.array([int(x)-int(y) for x,y in zip(b.flat,d.flat)],np.int32).reshape(b.shape)
        t=np.array([int(x)-int(y) for x,y in zip(s.flat,b.flat)],np.int32).reshape(b.shape)
        p,n=split_signed_codes(r);tp,tn=split_signed_codes(t)
        actual=compose_layer_codes(d,p,n,tp,tn)
        np.testing.assert_array_equal(actual,s)
        assert not np.shares_memory(actual,s)


@pytest.mark.parametrize('operation',[0,1,2,3])
def test_each_intermediate_clipping_is_rejected_even_if_later_cancelled(operation):
    arrays=[np.zeros((1,1,3),np.uint16) for _ in range(5)]
    # API order D,P,N,TP,TN; arithmetic order D-N+P-TN+TP.
    if operation==0:arrays[2][:]=1;arrays[1][:]=1
    if operation==1:arrays[0][:]=65535;arrays[1][:]=1;arrays[4][:]=1
    if operation==2:arrays[4][:]=1;arrays[3][:]=1
    if operation==3:arrays[0][:]=65535;arrays[3][:]=1
    with pytest.raises(ValueError):compose_layer_codes(*arrays)


@pytest.mark.parametrize('value',[-2147483648,-65536,65536,2147483647])
def test_full_int32_out_of_range_never_wraps(value):
    with pytest.raises(ValueError):split_signed_codes(np.full((1,1,3),value,np.int32))


def test_max_width_and_endian_strides():
    d=np.full((1,65536,3),20000,dtype='>u2')[:,::-1]
    p=np.full(d.shape,2000,dtype='<u2');n=np.full(d.shape,1000,dtype='>u2')
    out=compose_layer_codes(d,p,n);assert np.all(out==21000)
    with pytest.raises(ValueError):split_signed_codes(np.zeros((1,65537,3),np.int32))
