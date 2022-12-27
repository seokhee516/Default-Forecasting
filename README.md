# 금융권 대출 상환 이진분류 예측
![piggy-3610444_960_720](https://user-images.githubusercontent.com/86893209/181442606-f106337c-57b8-4a24-ab14-18271bf3bdaa.jpg)
# 프로젝트 개요

- 머신러닝을 통해 **대출 상환 여부를 이진분류 예측**하는 프로젝트입니다.
    - 금융권에서 대출 진행 시 금융거래 이력을 기반으로 산정하게 됩니다. 따라서 금융 거래 이력이 없는 사회초년생이나 주부와 같은 ‘씬 파일러(Thin Filer)’ 경우 대출을 갚을 능력이 있더라도 불이익을 겪게됩니다. 이를 보완해보고자 머신러닝을 통해 대출 상환 상태를 예측해보고, 어떤 특징이 예측에 가장 많은 영향을 주는지 기여도를 확인해보았습니다.

# 프로젝트 목표

### 대출 상환 상태를 정상(상환, 유지)과 불량(연체, 회수불능)으로 이진 분류 문제 예측

# 문제 상황 해결 과정

### 사용된 기술

- `Pandas` - 데이터 전처리 및 EDA
- `SMOTE` - 클래스 불균형 조절을 위한 OverSampling
- `Logistic Regreesion`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, `Catboost` - 이진 분류 예측
- `SHAP` - 특성 기여도 계산

# 프로젝트 진행 과정

## 데이터셋 설명

- `Lending Club 2007-2020Q3` - Lending Club(렌딩 클럽)은 미국 유명 **P2P 대출 업체** 입니다.
을 하고 있습니다.  캐글에서 제공하고 있는 데이터세트의 **2019년 데이터를 활용**하여 **2020년 상환상태를 예측**하였습니다.
- 캐글 주소 - [https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)

## EDA 및 데이터 전처리

### 140개 변수들 중 크게 다섯가지로 구분하여 총 30가지 변수 선정

- 대출자 정보, 렌딩클럽 내부정보, 금융계좌 정보, 카드정보, 연체정보로 다섯가지 범주로 구분하였습니다.
- 데이터 상세
    
    ![Untitled (1)](https://user-images.githubusercontent.com/86893209/183793641-175504e6-2ac2-4bbe-a8eb-adddfa1e6afa.png)
    

### 타겟변수의 분포가 정상 상환에 치우쳐진 불균형한 데이터라는 것을 확인

- 다음으로 모델검정을 위해 2020년 데이터를 테스트세트를 지정하였습니다. 나누어진 데이터의 범주를 묶거나 타입을 변형하는 방법으로 전처리를 진행하였습니다.

![Untitled (2)](https://user-images.githubusercontent.com/86893209/183793700-4fb97b82-2fdc-4b27-a337-8ca725e4ef45.png)

0: 정상 1: 불량

### 분석을 위해 데이터를 정리하고 Train, Validation, Test 세트로 나눔

- 범주형 변수 차원 증가 방지를 위해 범주군으로 묶고, 불필요한 기호를 제거하고, 형 변환 과정을 진행하였습니다. 또한 데이터 세트를 나눠주었습니다.

## 모델링

### 평가지표로 Accuracy, Precision, Recall, F1_Score 사용
### SMOTE를 이용한 Oversampling

![Untitled (3)](https://user-images.githubusercontent.com/86893209/183793719-85930add-b96f-45fd-8ee7-18acfd770b74.png)

(전) 정상 라벨의 수: 124437, 불량 라벨의 수: 4420

![Untitled (4)](https://user-images.githubusercontent.com/86893209/183793733-70b087c6-f252-40c1-9b55-5728f2ad9402.png)

(후) 정상 라벨의 수: 124437, 불량 라벨의 수: 62218

### Logistic Regreesion, Decision Tree, Random Forest, XGBoost, LightGBM, Catboost 모델링
|  | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regreesion | 80.6% | 17.9% | 22.8% | 20.0% |
| Logistic Regreesion(Oversampling) | 65.3% | 45.8% | 22.0% | 29.7% |
| Decision Tree | 41.2% | 14.0% | 87.4% | 24.1% |
| Decision Tree(Oversampling) | 74.8% | 57.6% | 92.4% | 71.0% |
| Random Forest | 62.4% | 21.8% | 97.5% | 35.7% |
| Random Forest(Oversampling) | 89.8% | 78.3% | 96.3% | 86.3% |
| XGBoost | 78.2% | 32.5% | 96.5% | 48.6% |
| XGBoost(Oversampling) | 88.7% | 76.0% | 96.7% | 85.1% |
| LightGBM | 35.3% | 13.6% | 94.5% | 23.8% |
| LightGBM(Oversampling) | 58.5% | 44.4% | 97.9% | 61.1% |
| Catboost | 35.8% | 13.5% | 92.6% | 23.5% |
| Catboost(Oversampling) | 96.2% | 78.3% | 63.9% | 70.4% |
## 결과 해석

![Untitled (5)](https://user-images.githubusercontent.com/86893209/183793755-48d24220-968c-4707-8cd8-4ec27488fb9c.png)

- Feature 0: 대출자의 연수입
- Feature 9: 월상환액
- Feature 24: 전체 금액 대비 변제 금액
- Feature 5: 차용자의 마지막 FICO가 속한 하위 경계 범위
- Feature 26: 상환된 총 원금액
- Feature 27: 상환된 총 이자금액

### 대출을 상환한 고객 데이터의 SHAP 밸류를 계산하여 어떤 특징이 기여도가 큰지 확인

- 대출자 **연수입**이 기여도가 가장 큼
- 그 외 **월상환액과 전체 금액 대비 상환된 금액**의 기여도가 큼

# 결론

모델 해석 결과 월 소득의 기여도가 가장 높았고, **월 상환액과 상환된 변제 금액** 또한 높은 기여도를 차지한다.

### → ‘거치식 상환’ 대비 ‘원(리)금 상환’과 같은 납부 방식이 대출에 중요한 영향을 미친다

# 배운 점

### 불균형한 데이터에 오버샘플링을 시도함

- 일반적인 문제를 풀어내는 다른 데이터들과 달리, 대출 데이터는 대출을 불이행하는 사람의 숫자가 매우 적기 때문에 데이터가 불균형했습니다. 이에 부족한 데이터를 늘려주기 위한 오버샘플링 기법을 시도하고, 성능을 개선시켰습니다.

### 금융업 머신러닝 모델의 의미를 해석하고 인사이트를 찾아냄

- 금융권에서는 정확성을 높이는 것은 물론, 결과에 대한 판단 근거를 전문가들이 이해할 수 있도록 설명가능한 AI의 중요성에 대해 배웠습니다. 이에 SHAP 밸류를 사용하여 직접 적용해볼 수 있었습니다.

# 참고

김은미, & 박지영. (2018, May). Lending Club 데이터를 이용한 다분류 기반의 개인신용등급 예측. In KMIS International Conference (pp. 633-637).

Sun, X. (2020, September). Prediction of the Borrowers' Payback to the Loan with Lending Club Data. In 2020 International Conference on Modern Education and Information Management (ICMEIM) (pp. 375-379). IEEE.

이군희, 유영범, & 하승인. (2017). 개인신용평가 모형을 위한 딥러닝 활용에 대한 연구. 대한산업공학회 춘계공동학술대회 논문집, 4042-4047.
