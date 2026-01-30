import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD DATA
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# 2. CLEANING
remove = ['Unnamed: 0', 'id']
df_train = df_train.drop(remove, axis=1)
df_test = df_test.drop(remove, axis=1)

# 3. PREPROCESSING
category_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in category_cols:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')

# Define X and y
x_train = df_train.drop('satisfaction', axis=1)
y_train = df_train['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

x_test = df_test.drop('satisfaction', axis=1)
y_test = df_test['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# 4. CREATE DATASET OBJECT
data_train = lgb.Dataset(x_train, label=y_train, categorical_feature='auto')
data_test = lgb.Dataset(x_test, label=y_test, reference=data_train, categorical_feature='auto')

# 5. TRAIN MODEL
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

model = lgb.train(params, data_train, num_boost_round=100)

# 6. PREDICTION 
y_pred_prob = model.predict(x_test)

y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred_prob]

# 7. EVALUATION
print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred_binary) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

lgb.plot_importance(model, max_num_features=10, title='Key Drivers')
plt.savefig('lgbm_feature_importance.png')
plt.show()