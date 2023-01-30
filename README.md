# 대출 상환 이진분류 예측

## 개요

머신러닝을 통해 **대출 상환 여부를 이진분류 예측**하는 프로젝트입니다.
- 금융권에서 대출 진행 시 금융거래 이력을 기반으로 산정하게 됩니다. 따라서 금융 거래 이력이 없는 사회초년생이나 주부와 같은 ‘씬 파일러(Thin Filer)’ 경우 대출을 갚을 능력이 있더라도 불이익을 겪게됩니다. 이를 보완해보고자 머신러닝을 통해 대출 상환 상태를 예측해보고, 어떤 특징이 예측에 가장 많은 영향을 주는지 기여도를 확인해보았습니다.

## 기술 요소
- `Python`
### Database
- `PostgreSQL`
### Model Development
-  Data Preprocessing & Data analysis
    - `Pandas` : 데이터 전처리 및 EDA
    - `SMOTE` : 클래스 불균형 조절을 위한 OverSampling
    - `SHAP` : 특성 기여도 계산 모델 결과 해석
-  Modeling
    - `Scikit-learn`
    - `Logistic Regreesion`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, `Catboost` : 이진 분류 예측 Machine Learning Model
    - `RandomizedSearchCV` : Hyper Parameter Tuning

### Model Deployment
- `FastAPI`
- `Streamlit` : webapp 화면 구성


## 프로젝트 구성


## Version
### Version 1.0.0
- 2021.11 ~ 2021.12
- Data Extraction and analysis
- Machine Learning Model Development
### Version 2.0.0
- 2023.01 ~ 2023.01
- Save data in Database
- Prediction Service API Serving

## 사용방법 
1. fastapi 실행
    ```
    python -m app
    ```
2. streamlit 실행
    ```
    cd app/views
    streamlit run main_views.py --server.port 30002
    ```
