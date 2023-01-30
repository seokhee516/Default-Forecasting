import streamlit as st
import streamlit.components.v1 as components
from collections import OrderedDict
import os
import pickle
import json
import datetime
import requests
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

url = 'http://localhost:7504'
predict_endpoint = '/model/predict/'
shap_endpoint = '/model/shap/'

def get_user_input_features():
    """
    메뉴 및 기타 input 값을 받기 위한 함수
    :return: json 형식의 user input 데이터
    """

    user_features = {"menu_name": st.sidebar.selectbox('Menu', ['predict'])}

    return [user_features]


def get_raw_input_features():
    """
    raw data input 값을 받기 위한 함수
    :return: json 형식의 raw input 데이터
    """
    raw_features = {"AP": st.sidebar.slider('Atomospheric Pressure', 990.0, 1040.0, (990.0 + 1040.0)/2),
                    "AT": st.sidebar.slider('Ambient Temperature', 1.0, 38.0, (1.0 + 38.0)/2),
                    "RH": st.sidebar.slider('Relative Humidity', 20.0, 105.0, (20.0 + 105.0)/2),
                    "V": st.sidebar.slider('Vaccum', 25.0, 85.0, (25.0 + 85.0)/2)
                    }

    return [raw_features]

def draw_shap_plot(base_value, shap_values, data, height=None):
    """
    shap plot 을 streamlit 어플리케이션 상에 표시하기 위한 함수
    :param data: 예측 결과값을 제외하고 사용된 변수 값만 포함하는 데이터
    :param base_value: shap 기준값
    :param shap_values: 변수별 shap 값
    :param height: 그림 height
    :return: None
    """
    p = shap.force_plot(base_value, shap_values, data)
    shap_html = f"<head>{shap.getjs()}</head><body>{p.html()}</body>"
    components.html(shap_html, height=height)

def streamlit_main():
    st.title("대출 상환 이진분류 예측")
    st.sidebar.header('User Menu')
    user_input_data = get_user_input_features()

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()

    submit = st.sidebar.button('예측하기')
    if submit:
        response = requests.post(url + predict_endpoint)
        results = response.text.strip('\"')
        st.subheader("결과")
        st.write(f'Prediction: {results}')
        
        # expander 형식으로 model input 표시
        st.subheader('Input Features')
        features_selected = ['AT', 'V', 'AP', 'RH']
        
        model_input_expander = st.expander('Model Input')
        model_input_expander.write('Input Features: ')
        model_input_expander.text(", ".join(list(raw_input_data[0].keys())))
        model_input_expander.json(raw_input_data[0])
        model_input_expander.write('Selected Features: ')
        model_input_expander.text(", ".join(features_selected))
        # selected_features_values = OrderedDict((k, results[k]) for k in features_selected)
        # model_input_expander.json(selected_features_values)
    
if __name__ == '__main__':
    streamlit_main()