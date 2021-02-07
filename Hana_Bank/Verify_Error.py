import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
from dataset import Dataset

pd.set_option('display.max_colwidth', 300)
pd.set_option('display.max_columns', 300)


def fund_check(data_verify):
    """
    펀드상품에서 잘 매핑이 됬는지 확인
    """
    fund_code_name = Dataset.check_condition_code_explanation('펀드번호', '펀드상품명', data_frame=data_verify.goods_fund)
    fund_name_code = Dataset.check_condition_code_explanation('펀드상품명', '펀드번호', data_frame=data_verify.goods_fund)
    check_list = [fund_code_name, fund_name_code]
    for df in check_list:
        if df.empty:
            continue
        else:
            print(df)


def factory_check(data_verify):
    """
    수신_여신상품 매핑 확인
    """
    code_name = Dataset.check_condition_code_explanation('상품코드', '상품명', data_frame=data_verify.goods_factory)
    name_code = Dataset.check_condition_code_explanation('상품명', '상품코드', data_frame=data_verify.goods_factory)
    detailed_condition_code_name = Dataset.check_condition_code_explanation('상세조건관리번호', '상세조건설명',
                                                                            data_frame=data_verify.goods_factory)
    name_detailed_condition_code = Dataset.check_condition_code_explanation('상세조건설명', '상세조건관리번호',
                                                                            data_frame=data_verify.goods_factory)
    condition_code_name = Dataset.check_condition_code_explanation('조건코드', '조건설명', data_frame=data_verify.goods_factory)
    name_condition_code = Dataset.check_condition_code_explanation('조건설명', '조건코드', data_frame=data_verify.goods_factory)
    check_list = [code_name, name_code, detailed_condition_code_name, name_detailed_condition_code, condition_code_name, name_condition_code]

    for df in check_list:
        if df.empty:
            continue
        else:
            print(df)


def bank_assurance_check(data_verify):
    """
    Key : (은행상품명, 보험사상품명) Value : 은행상품코드
    error에서 출력이 되면 문제
    """
    bank_code_goodsname_assurancename = {}
    error = []
    for index, value in data_verify.bank_assurance_data.iterrows():
        bank_key = (value['은행상품명'], value['보험사상품명'])
        if bank_key not in bank_code_goodsname_assurancename.keys():
            bank_code_goodsname_assurancename[bank_key] = value['은행상품코드']
        else:
            error.append(bank_key)
    if error:
        print(error)


if __name__=='__main__':
    """
    이 파일은 단순히 모든 데이터들이 잘 매핑되어있는지 확인하는데 목적이 있다.
    여기서도 역시 data path는 개인에 맞게 실행하면 된다.
    """
    data_verify = Dataset(data_path='/home/kibum/recommender_system/Hana_Bank/Modify_Hana_Data.xlsx')
    data_verify.load_data(factory_sheet='수신_여신상품(상품팩토리)', fund_sheet='펀드상품(펀드상품기본)', bank_assurance='방카상품(방카슈랑스기본)')
    # 1406
    len(data_verify.goods_factory['상품코드'].unique())
    len(data_verify.goods_factory['상품명'].unique())
    # Nan 확인
    print(data_verify.goods_fund.isnull().sum())
    # 수신 여신상품 정보
    factory_check(data_verify)
    # 펀드상품
    fund_check(data_verify)
    # 방카상품 매핑 확인
    bank_assurance_check(data_verify)


