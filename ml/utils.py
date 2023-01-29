import pandas as pd
def engineer(df):
    '''
    feature engineering 함수
    '''
    df = df.copy()

    # emp_length 범주형 변수 차원 증가 방지를 위해 범주군으로 묶음
    df.loc[(df['emp_length']== '< 1 year')|(df['emp_length']== '1 year')|
           (df['emp_length']== '2 years')|(df['emp_length']== '3 years'),'emp_length'] = 'less than 3 years'
    df.loc[(df['emp_length']== '4 years')|(df['emp_length']== '5 years')|
           (df['emp_length']== '6 years')|(df['emp_length']== '7 years')|
           (df['emp_length']== '8 years')|(df['emp_length']== '9 years'),'emp_length'] = 'more than 4 years and less than 9 years'
    df['emp_length'].fillna('unemployed', inplace=True)

    # purpose 범주형 변수 차원 증가 방지를 위해 범주군으로 묶음
    df.loc[(df['purpose']== 'car')|(df['purpose']== 'home_improvement')|
           (df['purpose']== 'house')|(df['purpose']== 'major_purchase')|
       (df['purpose']== 'medical')|(df['purpose']== 'moving')|
       (df['purpose']== 'other')|(df['purpose']== 'renewable_energy')|
       (df['purpose']== 'small_business')|(df['purpose']== 'vacation')|(df['purpose']== 'wedding'),'purpose'] = 'General loan debt'
    # int_rate % 기호를 제거하고 수치형으로 변환
    df['int_rate'] = df['int_rate'].replace('%','', regex=True).apply(pd.to_numeric)

    # sub_grade 수치형으로 변환
    sub_grade_ranks = {'A1': 1.1, 'A2': 1.2, 'A3': 1.3, 'A4': 1.4, 'A5': 1.5, 
                       'B1': 2.1, 'B2': 2.2, 'B3': 2.3, 'B4': 2.4, 'B5': 2.5, 
                       'C1': 3.1, 'C2': 3.2, 'C3': 3.3, 'C4': 3.4, 'C5': 3.5, 
                       'D1': 4.1, 'D2': 4.2, 'D3': 4.3, 'D4': 4.4, 'D5': 4.5}
    df['sub_grade'] = df['sub_grade'].map(sub_grade_ranks)


    # Reset index
    df = df.reset_index(drop=True)

    return df
