import pandas as pd
from timeit import default_timer as timer


class Dataset(object):
    def __init__(self, data_path='/home/kibum/recommender_system/Hana_Bank/Hana_Data.xlsx'):
        """
        goods_factory : 수신여신상품 데이터
        goods_fund : 펀드상품 데이터
        bank_assurance_data : 방카상품 데이터
        detailed_condition_code_explanation_dict : 수신 여신상품에서  상세조건관리번호와 설명 매핑
                                                key - 상세조건설명, value - 상세조건관리번호 List
        modify_detailed_code_explanation : 수신 여신상품에서 수정된 매핑정보
        fund_goods_null_count : 펀드상품에서 각 열별로 Nan의 개수
        fund_goods_problem_code_explanation : 문제있는 펀드상품명과 펀드번호의 DataFrame
        fund_goods_modify_code_explanation : 수정된 펀드상품명과 펀드번호의 DataFrame
        bank_assurance_null_count : 방카상품에서 각 열별로 Nan의 개수
        bank_assurance_modify_column : 제거하고 고려할 Column
        bank_assurance_problem_bankname_code : 방카상품에서 문제의 은행상품명과 은행상품코드 데이터프레임
        bank_assurance_modify_bankname_code : 방카상품에서의 수정한 데이터프레임(은행상품코드, 은행상품명, 보험사상품명)
       """
        self.data_path = data_path
        self.goods_factory = None
        self.goods_fund = None
        self.bank_assurance_data = None
        self.detailed_condition_code_explanation_dict = {}
        self.modify_detailed_code_explanation = None
        self.fund_goods_null_count = None
        self.fund_goods_problem_code_explanation = None
        self.fund_goods_modify_code_explanation = None
        self.bank_assurance_null_count = None
        self.bank_assurance_modify_column = None
        self.bank_assurance_problem_bankname_code = None
        self.bank_assurance_modify_bankname_code =None

    def load_data(self, factory_sheet=None, fund_sheet=None, bank_assurance=None):
        """
        데이터 로드
        """
        if factory_sheet is not None:
            goods_factory = pd.read_excel(self.data_path, sheet_name=factory_sheet)
            self.goods_factory = goods_factory
        if fund_sheet is not None:
            goods_fund = pd.read_excel(self.data_path, sheet_name=fund_sheet)
            self.goods_fund = goods_fund
        if bank_assurance is not None:
            bank_assurance_data = pd.read_excel(self.data_path, sheet_name=bank_assurance)
            self.bank_assurance_data = bank_assurance_data

    def preprocess_code_name(self):
        """
        수신 여신상품에서 상품코드와 상품명 전처리 - 문제 있음
        2개 문제 직접 해결 - 상품코드 : 271028000101, 0380019W01301
        """
        code_name_dict = {}
        error_index = []
        correct_index = []
        code_name = pd.concat([self.goods_factory['상품코드'], self.goods_factory['상품명']], axis=1)
        for index, value in code_name.iterrows():
            if value['상품명'] not in code_name_dict.keys():
                code_name_dict[value['상품명']] = value['상품코드']
            else:
                if value['상품코드'] != code_name_dict[value['상품명']]:
                    error_index.append([index, value['상품코드'], value['상품명']])
                    correct_index.append([index, code_name_dict[value['상품명']]])
        # 처리 시간 : 18.3268초
        self.goods_factory.drop(self.goods_factory.index[self.goods_factory['상품코드'] == 271028000101], inplace=True)
        self.goods_factory.drop(self.goods_factory.index[self.goods_factory['상품코드'] == '0380019W01301'], inplace=True)

    # start end date check
    def check_start_end_date(self):
        """
        상품명과 판매시작일, 판매종료일이 잘 매핑되어있는지 확인 - 문제 없음
        """
        start_last_date_dict = {}
        error_index = []
        correct_index = []
        start_last_date = self.goods_factory[['상품명', '판매시작일자', '판매종료일자']]
        for index, value in start_last_date.iterrows():
            if value['상품명'] not in start_last_date_dict.keys():
                start_last_date_dict[value['상품명']] = [value['판매시작일자'], value['판매종료일자']]
            else:
                if value['판매시작일자'] != start_last_date_dict[value['상품명']][0] or value['판매종료일자'] != \
                        start_last_date_dict[value['상품명']][1]:
                    error_index.append([index, value['상품명'], value['판매시작일자'], value['판매종료일자']])
                    correct_index.append(
                        [index, start_last_date_dict[value['상품명']][0], start_last_date_dict[value['상품명']][1]])

    @staticmethod
    def check_condition_code_explanation(key_code, value_word, data_frame):
        """
        매핑이 잘 되어있는지 확인하는 함수
        매핑이 1:1로 잘 되어있는 경우 데이터프레임에 아무 값도 없음
        매핑이 1:n으로 되어있는 경우 데이터프레임에 문제의 튜플이 담기게 됨
        """
        check_dict = {}
        error_index = []
        correct_index = []
        condition_code_explain = data_frame[[key_code, value_word]]
        # 여기서 조건코드와 조건설명을 바꿔서도 Check
        for index, line in condition_code_explain.iterrows():
            if line[key_code] not in check_dict.keys():
                check_dict[line[key_code]] = line[value_word]
            else:
                if line[value_word] != check_dict[line[key_code]]:
                    error_index.append([index, line[key_code], line[value_word]])
                    correct_index.append([line[key_code], check_dict[line[key_code]]])
        return pd.concat([pd.DataFrame(error_index, columns=['index', key_code + '_err', value_word + '_err']),
                          pd.DataFrame(correct_index, columns=[key_code + '_cor', value_word + '_cor'])], axis=1)

    def mapping(self, df):
        """
        람다 함수에 이용(apply)
        """
        self.detailed_condition_code_explanation_dict[df.name] = set(df['상세조건관리번호'].tolist())

    def mapping_modify_detailed_code_explanation(self):
        """
        상세조건과 상세조건코드 매핑이 안되어있는 것을 수정하는 코드
        modify_detailed_code_explanation : 상세조건설명, 상세조건관리번호 순으로 저장(수정본)
        """
        modify_detailed_code_explanation = []
        self.goods_factory.groupby(by=['상세조건설명']).apply(self.mapping)
        for key, value in self.detailed_condition_code_explanation_dict.items():
            detailed_name = value.pop()
            modify_detailed_code_explanation.append([key, detailed_name])
            self.goods_factory.loc[self.goods_factory['상세조건설명'] == key, '상세조건관리번호'] = detailed_name
        self.modify_detailed_code_explanation = pd.DataFrame(modify_detailed_code_explanation, columns=['상세조건설명', '상세조건관리번호'])
