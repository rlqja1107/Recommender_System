import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
from dataset import Dataset


def execute_fund_goods(data):
    """
    펀드상품(펀드상품기본) 전처리
    """
    # 데이터 Load
    data.load_data(fund_sheet='펀드상품(펀드상품기본)')

    # Nan 확인 / return
    data.fund_goods_null_count = data.goods_fund.isnull().sum()
    # Nan이 12076개로 제거
    del data.goods_fund['펀드투자성향구분']
    # 하나의 펀드번호에 펀드상품명은 잘 매핑, 그러나 펀드상품명에 펀드번호는 Mapping 문제
    # fund_num_goods 는 비어있음 -> 정상
    fund_num_goods = Dataset.check_condition_code_explanation('펀드번호', '펀드상품명', data_frame=data.goods_fund)

    # fund_goods_problem_code_explanation 은 비어있지 않음 -> 수정필요
    data.fund_goods_problem_code_explanation = Dataset.check_condition_code_explanation('펀드상품명', '펀드번호', data_frame=data.goods_fund)
    del data.fund_goods_problem_code_explanation['펀드상품명_cor']
    
    # 아래 코드를 통해 중복값이 9개 존재임을 알 수 있음
    # len(data.fund_goods_problem_code_explanation['펀드상품명_err'].unique())
    # len(data.fund_goods_problem_code_explanation['펀드상품명_err'])

    # 펀드상품명에 여러 개의 펀드번호 매핑되어있는 문제 해결
    modify_frame = solve_fund_name_num_mapping(data.fund_goods_problem_code_explanation, data)

    # 펀드번호와 펀드상품명 매칭 정보 보기
    data.fund_goods_modify_code_explanation = pd.DataFrame(modify_frame, columns=['펀드번호', '펀드상품명'])

    # 아래의 코드를 통해 모든 펀드상품이 Unique하다는 것을 알 수 있음
    print(len(data.goods_fund['펀드상품명']))
    print(len(data.goods_fund['펀드상품명'].unique()))

    # 아래 코드를 통해 잘 매핑되어있음을 알 수 있음
    print(len(data.goods_fund['펀드번호'].unique()))
    print(len(data.goods_fund['펀드상품명'].unique()))

    # 또는 아래 코드를 통해 문제가 없음을 확인, 비어있으면 제대로 수정 - 수정확인
    # problem_frame = Dataset.check_condition_code_explanation('펀드상품명', '펀드번호', data_frame = data.goods_fund)
    # del problem_frame['펀드상품명_cor']
    # print(problem_frame)

def information_print(data):
    """
    단순히 정보 출력 함수
    """
    # 펀드번호의 갯수 : 15809
    print("펀드번호 개수 :", len(data.goods_fund['펀드번호'].unique()))
    
    # 펀드상품명의 갯수 : 15746
    print("펀드상품명의 개수 :", len(data.goods_fund['펀드상품명'].unique()))

    # Distinct한 운용사 개수 - 96
    print('운용사 개수 ', len(data.goods_fund['운용사명'].unique()))

def solve_fund_name_num_mapping(problem_frame, data):
    fund_name_num_list = []
    modify_frame = []
    for index, value in problem_frame.iterrows():
        if value['펀드상품명_err'] not in fund_name_num_list:
            # problem_frame 중 중복 펀드상품명이 있기 때문에 이 중 하나만 처리
            fund_name_num_list.append(value['펀드상품명_err'])
            temp_frame = data.goods_fund.loc[data.goods_fund['펀드상품명'] == value['펀드상품명_err']]
            num_list = temp_frame['펀드번호'].tolist()
            modify_funds_num = num_list.pop()
            data.goods_fund.drop(data.goods_fund.index[data.goods_fund['펀드번호'].isin(num_list)], inplace=True)
            modify_frame.append([modify_funds_num, value['펀드상품명_err']])
    return modify_frame
