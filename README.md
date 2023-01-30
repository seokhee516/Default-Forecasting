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

## 주요 기능
- 대출자의 정보 입력하면 미상환 확률 예측
- 예측에 영향을 준 요인 시각화

## 프로젝트 구성

```
├── app
|   ├── __main__.py
|   ├── main.py                             - fastapi backend
|   ├── models
|   |   ├── preprocessing_objects.pkl       - 전처리 파일
|   |   └── rf_model.joblib                 - Random Forest 모델 파일
|   └── views
|       └── main_views.py                   - stramlit frontend
├── data
|   ├── data.csv                            - 실험용 data
|   └── make_dataset.py                     - raw 데이터를 DB에 저장
├── ml
|   ├── predict_model.py                    - 저장된 모델 검증
|   ├── train_model.py                      - 모델 train 및 저장
|   └── utils.py                            - 전처리 함수
└── notebooks
    ├── README.md
    └── experiment.ipynb                    - EDA 및 모델 실험
```
## Version
### Version 1.0.0
- 2021.11 ~ 2021.12
- Model Development
- `notebooks/experiment.ipynb`에 실험 과정 및 결과 기록
### Version 2.0.0 (진행 중)
- 2023.01 ~ 2023.01
- PostgreSQL Database
- Model Deployment

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
