import qtensor.utils as utils

def test_mergeable_indices():
    buckets = [
        ['ab', 'ad']
        ,['bc', 'bd']
        ,['cd']
        ,['df']
        ,[]
    ]
    peo = 'abcdf'
    merged_ix, width = utils.find_mergeable_indices(peo, buckets)
    print(f'merged_ix={merged_ix}')
    print(f'width={width}')
    assert len(merged_ix) > 0
    assert [3,4] in merged_ix

    buckets = [
         ['ŵƴĴŎžĨřùķŁťƛƋąĆšīŜúĎĊŃūďĪĲƎŚǅ', 'ŵƴĴŎžūƃƜƟƖŧǀƈƵƿǍŬƇƧƼǈƯĲƎŚǅ', 'ŵƎ']
        ,['žƴ' 'ƴ', 'ƴǍ']
        ,['ĲĴ']
        ,['ŎŚ']
        ,['šžƀƖǍ']
    ] + [[]]*60
    peo = 'ŵƴĴŎžƀŎžĨřùķŁťƛƋąĆšīŜúĎĊŃūďĪĲƎŚǅƃƜƟƖŧǀƈƵƿǍŬƇƧƼǈƯ'
    merged_ix, width = utils.find_mergeable_indices(peo, buckets)
    print(f'merged_ix={merged_ix}')
    print(f'width={width}')
    assert len(merged_ix) > 0
    assert [0,1,2,3] in merged_ix
