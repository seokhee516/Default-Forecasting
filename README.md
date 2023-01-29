# 대출 상환 이진분류 예측

## 개요

머신러닝을 통해 **대출 상환 여부를 이진분류 예측**하는 프로젝트입니다.
- 금융권에서 대출 진행 시 금융거래 이력을 기반으로 산정하게 됩니다. 따라서 금융 거래 이력이 없는 사회초년생이나 주부와 같은 ‘씬 파일러(Thin Filer)’ 경우 대출을 갚을 능력이 있더라도 불이익을 겪게됩니다. 이를 보완해보고자 머신러닝을 통해 대출 상환 상태를 예측해보고, 어떤 특징이 예측에 가장 많은 영향을 주는지 기여도를 확인해보았습니다.

## 기술 요소
### python
### Data Preprocessing & Data analysis
- `Pandas` - 데이터 전처리 및 EDA
- `SMOTE` - 클래스 불균형 조절을 위한 OverSampling
- `SHAP` - 특성 기여도 계산 모델 결과 해석
### Modeling
- `Logistic Regreesion`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, `Catboost` - 이진 분류 예측 Machine Learning Model

### Database
- PostgreSQL
# 프로젝트 구성


