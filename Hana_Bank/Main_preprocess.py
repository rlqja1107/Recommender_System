import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
from timeit import default_timer as timer
from dataset import Dataset
from Goods_Factory import execute_goods_factory
from Fund_Goods import execute_fund_goods
from Bank_Assurance import execute_bank_assurance

# 출력 길이 설정
pd.set_option('display.max_colwidth', 300)
pd.set_option('display.max_columns', 300)


def convert_time(time):
    minute = int(time/60)
    sec = time % 60
    return str(minute) + '분' + str(sec) +'초'


if __name__ == '__main__':
    """
    이 파일을 실행하여 원래의 파일을 전처리하는데 목적이 있다.
    """
    start = timer()
    # 데이터 위치 지정
    data = Dataset(data_path='/home/kibum/recommender_system/Hana_Bank/Hana_Data.xlsx')
    # 수신_여신상품(상품팩토리) 전처리
    execute_goods_factory(data)
    # 펀드상품(펀드상품기본) 전처리
    execute_fund_goods(data)
    # 방카상품(방카슈랑스기본) 전처리
    execute_bank_assurance(data)

    # 저장경로는 자유, 서식정하기
    with pd.ExcelWriter('/home/kibum/recommender_system/Hana_Bank/Modify_Hana_Data.xlsx') as writer:
        data.goods_factory.to_excel(writer, sheet_name='수신_여신상품(상품팩토리)', index=False)
        data.goods_fund.to_excel(writer, sheet_name='펀드상품(펀드상품기본)', index=False)
        data.bank_assurance_data.to_excel(writer, sheet_name='방카상품(방카슈랑스기본)', index=False)
    print("Total Preprocessing Time : {}".format(convert_time(timer()-start)))

