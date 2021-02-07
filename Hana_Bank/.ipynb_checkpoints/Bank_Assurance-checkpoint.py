import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
from dataset import Dataset


def execute_bank_assurance(data):
    """
    방카상품 전처리 실행
    """
    # 데이터 로드
    data.load_data(bank_assurance='방카상품(방카슈랑스기본)')

    # 결측치 개수 구하기
    data.bank_assurance_null_count = data.bank_assurance_data.isnull().sum()

    # 그 중 2500개 이상 Nan인 Column 제거도 가능, 여기서는 Nan이 있으면 모두 제거
    upper_3000 = data.bank_assurance_null_count[data.bank_assurance_null_count > 0]
    for element in upper_3000.index:
        del data.bank_assurance_data[element]

    # 고려할 Column 출력
    # print(data.bank_assurance_data.columns)
    # print(len(data.bank_assurance_data['은행상품코드'].unique()))
    # print(len(data.bank_assurance_data['은행상품명'].unique()))

    error_frame = Dataset.check_condition_code_explanation('은행상품명', '은행상품코드', data_frame=data.bank_assurance_data)
    del error_frame['은행상품명_cor']
    data.bank_assurance_problem_bankname_code = error_frame

    # 하나의 은행상품명에 여러 은행상품코드가 있는 문제 수정하기
    modify_code_bank_assurance = solve_bank_name_num_mapping(data, error_frame)
    # 순서대로 은행상품코드, 은행상품명, 보험사상품명 저장

    # 매핑된 은행상품코드, 은행상품명, 보험사상품명
    mapping_data = pd.DataFrame(modify_code_bank_assurance, columns=['은행상품코드', '은행상품명', '보험사상품명'])
    data.bank_assurance_modify_bankname_code = mapping_data

    # 제대로 매핑됬는지 검사 - 제대로 매핑됬으면 error_frame이 비어있음 - > 해결
    # error_frame = Dataset.check_condition_code_explanation('은행상품명', '은행상품코드', data_frame=data_bank.bank_assurance_data)
    # del error_frame['은행상품명_cor']
    # print(error_frame)

    # 띄어쓰기만 다르므로 삭제
    data.bank_assurance_data.drop(data.bank_assurance_data.index[data.bank_assurance_data['은행상품명'] == '무배당교보First저축보험III'], inplace=True)


def mapping(df, bank_goods_name, bank_code_name_dict):
    """
    Lambda 함수를 위해 사용되는 코드
    """
    bank_code_name_dict[(df.name, bank_goods_name)] = set(df['은행상품코드'].tolist())


def fill_bank_code_dict(error_frame, data):
    """
    방카상품에서 문제 있는 은행상품과 상품코드를 딕셔너리에 넣는 코드
    bank_code_name_dict : Key - (보험사상품명, 은행상품명), Value - 은행사상품코드 List
    """
    bank_code_name_dict = {}
    bank_name = []
    for index, value in error_frame.iterrows():
        if value['은행상품명_err'] not in bank_name:
            bank_name.append(value['은행상품명_err'])
            temp_frame = data.bank_assurance_data[data.bank_assurance_data['은행상품명'] == value['은행상품명_err']]
            temp_frame.groupby(by=['보험사상품명']).apply(lambda df: mapping(df, value['은행상품명_err'],bank_code_name_dict))
    return bank_code_name_dict


def solve_bank_name_num_mapping(data, error_frame):
    """
    하나의 은행상품명에 여러 은행상품코드의 Mapping 문제 해결
    하나의 은행상품코드에는 하나의 은행상품명이 매핑되어있기 때문에 아래 코드에서 삭제 가능 가능
    """
    bank_code_name_dict = fill_bank_code_dict(error_frame, data)
    modify_code_bank_assurance = []
    for k, v in bank_code_name_dict.items():
        assurance_name = k[0]
        bank_goods_name = k[1]
        modify_key = v.pop()
        
        data.bank_assurance_data.drop(data.bank_assurance_data.index[data.bank_assurance_data['은행상품코드'].\
                                           isin(list(v))], inplace=True)
        modify_code_bank_assurance.append([modify_key, bank_goods_name, assurance_name])
    return modify_code_bank_assurance
