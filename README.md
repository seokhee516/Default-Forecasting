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
    
    ![Untitled (1)](https://user-images.githubusercontent.com/86893209/183793641-175504e6-2ac2-4bbe-a8eb-adddfa1e6afa.png)
    

### 타겟변수의 분포가 정상 상환에 치우쳐진 불균형한 데이터라는 것을 확인

- 다음으로 모델검정을 위해 2020년 데이터를 테스트세트를 지정하였습니다. 나누어진 데이터의 범주를 묶거나 타입을 변형하는 방법으로 전처리를 진행하였습니다.

![Untitled (2)](https://user-images.githubusercontent.com/86893209/183793700-4fb97b82-2fdc-4b27-a337-8ca725e4ef45.png)

0: 정상 1: 불량

### 분석을 위해 데이터를 정리하고 Train, Validation, Test 세트로 나눔

- 범주형 변수 차원 증가 방지를 위해 범주군으로 묶고, 불필요한 기호를 제거하고, 형 변환 과정을 진행하였습니다. 또한 데이터 세트를 나눠주었습니다.

## ✔️ 모델링

### 평가지표로 Accuracy, Precision, Recall, F1_Score 사용
### SMOTE를 이용한 Oversampling

![Untitled (3)](https://user-images.githubusercontent.com/86893209/183793719-85930add-b96f-45fd-8ee7-18acfd770b74.png)

(전) 정상 라벨의 수: 124437, 불량 라벨의 수: 4420

![Untitled (4)](https://user-images.githubusercontent.com/86893209/183793733-70b087c6-f252-40c1-9b55-5728f2ad9402.png)
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

# 4️⃣ 결론

모델 해석 결과 예상대로 월 소득의 기여도가 가장 높았으나, 예상 외로 **월 상환액과 상환된 변제 금액** 또한 높은 기여도를 차지한다.

### → 💡 따라서 성실히 상환해가는 성실도가 대출에 중요한 영향을 미친다.

# 5️⃣ 프로젝트의 한계

### 다중공선성 문제

- 변수를 선택하는 과정에서 다중공선성의 문제를 해결하지 못했습니다. 특히 대출의 원금과 이자액은 높은 상관관계를 가지고 있어 변수 간 독립적인 관계가 아니었습니다.

### **대안 신용평가 방안 제안의 어려움**

- 금융이력이 부족한 사람들의 특수성을 고려하여, 기존 금융정보가 아닌 비금융정보로 대안신용평가를 시도해보려 하였습니다. 그러나 개인정보 보호 등으로 데이터 수집이 어려웠고, 전통적인 금융 정보를 활용하였습니다.

# 6️⃣ 배운 점

### 시간과 컴퓨터 자원을 최대한 활용함

- Colab 환경에서 모델링과 하이퍼 파라미터 튜닝을 하기 위해 최대한 시간을 활용했습니다. 이를 위해 GridSearchCV보다는 RandomSearchCV를 사용하였고, 파라미터를 max_depth. min_samples_leaf, min_samples_split 등으로 한정하였습니다. 이를 통해 아쉬움은 남지만 주어진 시간에서는 최선의 결과물을 만들어냈습니다.

### 불균형한 데이터에 오버샘플링을 시도함

- 일반적인 문제를 풀어내는 다른 데이터들과 달리, 대출 데이터는 대출을 불이행하는 사람의 숫자가 매우 적기 때문에 데이터가 불균형했습니다. 이에 부족한 데이터를 늘려주기 위한 오버샘플링 기법을 시도하고, 성능을 개선시켰습니다.

### 금융업 머신러닝 모델의 의미를 해석하고 인사이트를 찾아냄

- 금융권에서는 정확성을 높이는 것은 물론, 결과에 대한 판단 근거를 전문가들이 이해할 수 있도록 설명가능한 AI의 중요성에 대해 배웠습니다. 이에 SHAP 밸류를 사용하여 직접 적용해볼 수 있었습니다.

# 7️⃣ 참고

[Lending Club 데이터를 이용한 다분류 기반의 개인신용등급 예측.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09e1cb80-48b4-433c-af99-a2e8d891fb35/Lending_Club______.pdf)

[Prediction_of_the_Borrowers_Payback_to_the_Loan_with_Lending_Club_Data.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/027ea01f-ead7-46c9-a5a0-0af2622ade42/Prediction_of_the_Borrowers_Payback_to_the_Loan_with_Lending_Club_Data.pdf)

[개인신용평가 모형을 위한 딥러닝 활용에 대한 연구.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ac52d140-e81f-4cea-9874-f39eaa9e318e/______.pdf)
