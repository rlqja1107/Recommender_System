import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
from dataset import Dataset

data = Dataset(data_path='/home/kibum/recommender_system/Hana_Bank/Modify_Hana_Data.xlsx')
data.load_data()
data.load_data(factory_sheet='수신_여신상품(상품팩토리)', fund_sheet='펀드상품(펀드상품기본)', bank_assurance='방카상품(방카슈랑스기본)')

count_detailed_explanation = data.goods_factory.groupby(by=['상세조건설명']).size()
# 상세조건설명의 Unique 수 : 43538
len(data.goods_factory['상세조건설명'].unique())
# 5개 이하의 상세조건 수 : 41792
len(count_detailed_explanation[count_detailed_explanation < 5])
# 타입변경
len(data.goods_factory)

data.goods_factory['상세조건설명'].str.startswith('과목코드')
# -로 relation이 가능한 것은 68404개
data.goods_factory['상세조건설명'].str.contains('-').sum()
data.goods_factory.loc[data.goods_factory['상세조건설명'].str.contains('-'), '상세조건설명']
data.goods_factory['상세조건설명'].str.contains('가능').sum()
# 전체 tuple의 수는 222291
len(data.goods_factory['상세조건설명'])
# '-'가 있는 것 출력해보기
data.goods_factory['상세조건설명'] = data.goods_factory['상세조건설명'].astype(str)
data.goods_factory.loc[data.goods_factory['상세조건설명'].str.contains('-'),'상세조건설명']

relation_count =data.goods_factory.loc[data.goods_factory['상세조건설명'].str.contains('-'),'상세조건설명'].str.findall('\w+-').tolist()
relation_set = set()
for i in relation_count:
    if len(i) > 0:
        val = i[0]
        relation_set.add(val)
print(len(relation_set))
