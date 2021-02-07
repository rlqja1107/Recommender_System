import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
import pandas as pd
from dataset import Dataset


def check_mapping(data):
    """
    단순히 상세조건관리번호와 상세조건설명이 잘 매핑되어있는지 확인하는 코드
    이 코드는 생략 가능
    """
    condition_code_explain_dict = {}
    error_index = []
    correct_index = []
    condition_code_explain = data.goods_factory[['상세조건관리번호', '상세조건설명']]
    # 여기서 조건코드와 조건설명을 바꿔서도 Check
    for index, value in condition_code_explain.iterrows():
        if value['상세조건설명'] not in condition_code_explain_dict.keys():
            condition_code_explain_dict[value['상세조건설명']] = value['상세조건관리번호']
        else:
            if value['상세조건관리번호'] != condition_code_explain_dict[value['상세조건설명']]:
                error_index.append([index, value['상세조건설명'], value['상세조건관리번호']])
                correct_index.append([index, value['상세조건설명'], condition_code_explain_dict[value['상세조건설명']]])
    con = pd.concat([pd.DataFrame(error_index,columns=['i','a','b']), pd.DataFrame(correct_index,columns=['i','c','d'])], axis=1)


def execute_goods_factory(data):
    """
    상품팩토리 전처리
    """
    # 데이터 로드
    data.load_data(factory_sheet='수신_여신상품(상품팩토리)')

    # 상품코드와 상품명 Error Check 코드 - 중복 문제 2개 존재
    data.preprocess_code_name()

    # 종료일과 시작일이 상품명과 동일한지 Check - 이상 없음
    data.check_start_end_date()

    # 상세조건관리번호와 상세조건설명이 매핑이 잘 되어있는지 Check
    code_explain = Dataset.check_condition_code_explanation(key_code='상세조건관리번호', value_word='상세조건설명', data_frame=data.goods_factory)

    # 상세조건관리번호와 상세조건설명 수정
    data.goods_factory.loc[data.goods_factory['상세조건설명'] == '생계형저축한도-3000만원까지', '상세조건관리번호'] = 'A0160004'

    # 반대로 상세조건 설명에 상세조건관리번호가 매핑이 잘 되어있는지 Check - 문제 있음
    code_explain_1 = Dataset.check_condition_code_explanation(key_code='상세조건설명', value_word='상세조건관리번호', data_frame=data.goods_factory)

    # 상세조건설명과 상세조건관리번호 코드 수정
    data.mapping_modify_detailed_code_explanation()

    # 단순히 상세조건설명과 상세조건관리번호가 잘 매핑되어있는지 확인하는 코드
    # check_mapping(data)



