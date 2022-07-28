# 금융권 대출 상환 이진분류 예측
![piggy-3610444_960_720](https://user-images.githubusercontent.com/86893209/181442606-f106337c-57b8-4a24-ab14-18271bf3bdaa.jpg)
# 📋 프로젝트 개요

- 머신러닝을 통해 **대출 상환 여부를 이진분류 예측**하는 프로젝트입니다.
    - 금융권에서 대출 진행 시 금융거래 이력을 기반으로 산정하게 됩니다. 따라서 금융 거래 이력이 없는 사회초년생이나 주부와 같은 ‘씬 파일러(Thin Filer)’ 경우 대출을 갚을 능력이 있더라도 불이익을 겪게됩니다. 이를 보완해보고자 머신러닝을 통해 직접 예측해보고, 어떤 특징이 예측에 가장 많은 영향을 주는지 기여도를 확인해보았습니다.

# 1️⃣ 프로젝트 목표

### 대출 상환 상태를 정상(상환, 유지)과 불량(연체, 회수불능)으로 이진 분류 문제 예측

# 2️⃣ 문제 상황 해결 과정

### 사용된 기술

- `Pandas` - 데이터 전처리 및 EDA
- `SMOTE` - 클래스 불균형 조절을 위한 OverSampling
- `Logistic Regreesion`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, `Catboost` - 이진 분류 예측
- `SHAP` - 특성 기여도 계산

# 3️⃣ 프로젝트 진행 과정

## ✔️ 데이터셋 설명

- `Lending Club 2007-2020Q3` **- Lending Club(렌딩 클럽)**은 미국 유명 **P2P 대출 업체** 입니다.
을 하고 있습니다.  캐글에서 제공하고 있는 데이터세트의 **2019년 데이터를 활용**하여 **2020년 상환상태를 예측**하였습니다.
- 캐글 주소 - [https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)

## ✔️ EDA 및 데이터 전처리

### 140개 변수들 중 크게 다섯가지로 구분하여 총 30가지 변수 선정

- 대출자 정보, 렌딩클럽 내부정보, 금융계좌 정보, 카드정보, 연체정보로 다섯가지 범주로 구분하였습니다.
- 📁 데이터 상세
    
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bd74c1d4-c692-4dbd-b5d5-7809164aa004/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220728%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220728T071134Z&X-Amz-Expires=86400&X-Amz-Signature=e60d221a538b6536d03df4802ca4c86b42fd9592a02da891cec3beb0e6ad44f5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)
    

### 타겟변수의 분포가 정상 상환에 치우쳐진 불균형한 데이터라는 것을 확인

- 다음으로 모델검정을 위해 2020년 데이터를 테스트세트를 지정하였습니다. 나누어진 데이터의 범주를 묶거나 타입을 변형하는 방법으로 전처리를 진행하였습니다.

![0: 정상 1: 불량](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2f7343e6-50d4-481c-9b34-afec802823ee/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220728%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220728T071108Z&X-Amz-Expires=86400&X-Amz-Signature=f62be04462f93de9e22ea2ef90e14f887fe338ad6dde7b98b1d627218d0d4aba&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

0: 정상 1: 불량

### 분석을 위해 데이터를 정리하고 Train, Validation, Test 세트로 나눔

- 범주형 변수 차원 증가 방지를 위해 범주군으로 묶고, 불필요한 기호를 제거하고, 형 변환 과정을 진행하였습니다. 또한 데이터 세트를 나눠주었습니다.

## ✔️ 모델링

### 평가지표로 Accuracy, Precision, Recall, F1_Score 사용
### SMOTE를 이용한 Oversampling

![(전) 정상 라벨의 수: 124437, 불량 라벨의 수: 4420](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eeeb6fc5-66bf-479e-93e9-428755b3ff61/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220728%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220728T071054Z&X-Amz-Expires=86400&X-Amz-Signature=59a0ab98ae9443142036495b2214cc6b59dc78527d5f41a1c9e6810d5994a430&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

(전) 정상 라벨의 수: 124437, 불량 라벨의 수: 4420

![(후) 정상 라벨의 수: 124437, 불량 라벨의 수: 62218](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a169f957-3ec8-4eec-a7d2-e5d5e5078a5d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220728%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220728T071028Z&X-Amz-Expires=86400&X-Amz-Signature=0556a894cee5695b330072ffc146999b381eea504414e78a89950a5f19a2c09d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

(후) 정상 라벨의 수: 124437, 불량 라벨의 수: 62218

### Logistic Regreesion, Decision Tree, Random Forest, XGBoost, LightGBM, Catboost 모델링
|  | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regreesion | 65.3% | 45.8% | 22.0% | 29.7% |
| Decision Tree | 74.8% | 57.6% | 92.4% | 71.0% |
| Random Forest | 89.8% | 78.3% | 96.3% | 86.3% |
| XGBoost | 88.7% | 76.0% | 96.7% | 85.1% |
| LightGBM | 58.5% | 44.4% | 97.9% | 61.1% |
| Catboost | 96.2% | 78.3% | 63.9% | 70.4% |

## ✔️ 결과 해석

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3382039f-2251-4139-ad18-b2d4355e3601/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220728%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220728T070958Z&X-Amz-Expires=86400&X-Amz-Signature=3a12b5d0f6f77695ba2594f2ca0376e04972cad5dfcb6d09777d62e5a32ac290&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

- Feature 0: 대출자의 연수입
- Feature 9: 월상환액
- Feature 24: 전체 금액 대비 변제 금액
- Feature 5: 차용자의 마지막 FICO가 속한 하위 경계 범위
- Feature 26: 상환된 총 원금액
- Feature 27: 상환된 총 이자금액

### 대출을 상환한 고객 데이터의 SHAP 밸류를 계산하여 어떤 특징이 기여도가 큰지 확인

- 대출자 **연수입**이 기여도가 가장 큼
- 그 외 **월상환액과 전체 금액 대비 상환된 금액**의 기여도가 큼

# 4️⃣ 결론

모델 해석 결과 예상대로 월 소득의 기여도가 가장 높았으나, 예상 외로 **월 상환액과 상환된 변제 금액** 또한 높은 기여도를 차지한다.

### → 💡 따라서 성실히 상환해가는 성실도가 대출에 중요한 영향을 미친다.

# 5️⃣ 프로젝트의 한계

### 다중공선성 문제

- 변수를 선택하는 과정에서 다중공선성의 문제를 해결하지 못했습니다. 특히 대출의 원금과 이자액은 높은 상관관계를 가지고 있어 변수 간 독립적인 관계가 아니었습니다.

### 시간과 컴퓨터 자원의 한계

- 많은 모델과 하이퍼 파라미터 튜닝 하면서, 모델링 결과를 체계적으로 정리하지 못하였습니다. 또한 Colab 환경의 한정된 자원으로 학습하다보니, 시간이 많이 소요되었고 효율적으로 학습하지 못해서 아쉬움이 남습니다.

### **대안 신용평가 방안 제안의 어려움**

- 금융이력이 부족한 사람들의 특수성을 고려하여, 기존 금융정보가 아닌 비금융정보로 대안신용평가를 시도해보려 하였습니다. 그러나 개인정보 보호 등으로 데이터 수집이 어려웠고, 전통적인 금융 정보를 활용하였습니다.

# 6️⃣ 배운 점

### 성능을 높이기 위해 다양한 머신러닝 모델과 하이퍼파라미터 튜닝을 시도함

- 불균형한 데이터를 오버샘플링해보고, 랜덤포레스트부터 XG부스트까지 다양한 모델을 적용하고 하이퍼파라미터 튜닝까지 머신러닝 엔지니어로서 업무를 경험해볼 수 있었습니다.

### 금융업 머신러닝 모델의 의미를 해석하고 인사이트를 찾아냄

- 금융권에서는 정확성을 높이는 것은 물론, 결과에 대한 판단 근거를 전문가들이 이해할 수 있도록 설명가능한 AI의 중요성에 대해 배웠습니다. 이에 SHAP 밸류를 사용하여 직접 적용해볼 수 있었습니다.
