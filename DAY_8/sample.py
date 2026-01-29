import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
df=pd.read_csv("Churn_Modelling.csv")
remove=['RowNumber','CustomerId','Surname']
for col in remove:
    df=df.drop(col,axis=1)

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Geography']=le.fit_transform(df['Geography'])


X=df.drop('Exited',axis=1)
y=df['Exited']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,eval_metric='logloss',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")  
print(cm)
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.savefig('confusion_matrix_heatmap.png')
plt.show()
