import joblib
import pickle
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils import engineer

# 1. get data
with open("secret_key.p", "rb") as file:    # postgre sql과 연결
    secret_key = pickle.load(file)
    db_connect = psycopg2.connect(
        host=secret_key["host"],
        database=secret_key["database"],
        user=secret_key["user"],
        password=secret_key["password"],
    )
df = pd.read_sql("SELECT * FROM lending ORDER BY id DESC", db_connect)      # db에서 data 가져오기

test = df[(df['issue_d'] == 'Sep-2020') | (df['issue_d'] == 'May-2020')]    # test set 만들기
train = df.drop(test.index)

train, val = train_test_split(train, train_size = 0.8, stratify=train['Target'], random_state=10)   # validation set 만들기

train = engineer(train) # feature engineering
val = engineer(val)
test = engineer(test)

features = train.drop(columns=["id", "Target"]).columns     # # feature selection
X_train = train[features]
y_train = train['Target']
X_valid = val[features]
y_valid = val['Target']
X_test = test[features]
y_test = test['Target']

preprocessor = make_pipeline(
    OrdinalEncoder(), 
    SimpleImputer(strategy='mean')
)
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

sm = SMOTE(random_state = 10, sampling_strategy=0.5)    # over sampling
X_train_sm, y_train_sm = sm.fit_resample(X_train_processed, y_train.ravel())
X_val_sm, y_val_sm = sm.fit_resample(X_val_processed, y_valid.ravel())

# 2. model development and train
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

param_distributions = { 
    'n_estimators': randint(50, 800), 
    'max_depth': [5, 7, 9, 12, 15], 
    'min_samples_leaf': randint(2, 20),
    'min_samples_split': randint(2, 20) 
}

search = RandomizedSearchCV(
    rfc, 
    param_distributions=param_distributions, 
    n_iter=5, 
    cv=3, 
    scoring='f1', 
    verbose=10, 
    return_train_score=True, 
    n_jobs=-1, 
    random_state=10
)
search.fit(X_train_sm, y_train_sm)
rf_model = search.best_estimator_

train_pred = rf_model.predict(X_train_sm)
valid_pred = rf_model.predict(X_val_sm)

train_acc = accuracy_score(y_true=y_train_sm, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
train_f1 = f1_score(y_true=y_train_sm, y_pred=train_pred)
valid_f1 = f1_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)
print("Train F1 Score :", train_f1)
print("Valid F1 Score :", valid_f1)

# 3. save model
joblib.dump(rf_model, "rf_model.joblib")

# 4. save data
df.to_csv("data.csv", index=False)

# 5. save preprocessing objects
preprocessing_objects = {}
preprocessing_objects['preprocessor'] = preprocessor
preprocessing_objects['sm'] = sm
preprocessing_objects['features'] = features

with open('preprocessing_objects.pkl', 'wb') as handle:
    pickle.dump(preprocessing_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)