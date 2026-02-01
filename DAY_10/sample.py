import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error,r2_score

def save_plot(name):
    plt.savefig(name)
    plt.show()

df=pd.read_csv("city_day.csv")

le=LabelEncoder()
df['City_code']=le.fit_transform(df["City"])

features=['PM2.5','PM10','NO2','CO','SO2','O3','City_code']
target='AQI'

df=df.dropna(subset=[target])
df[features]=df[features].fillna(df[features].median())

X=df[features]
y=df[target]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(n_estimators=100,max_depth=6,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

print(f"Baseline Accuracy (R2): {r2*100:.2f}%")
print(f"Baseline Error (MAE):   {mae:.2f}")

param_grid={
    'n_estimators':[100,200],
    'min_samples_split':[2,5,10],
    'max_depth':[10,20,None]
}

grid_search=GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(x_train,y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")

best_model=grid_search.best_estimator_ 
y_pred_best=best_model.predict(x_test)

r2_best=r2_score(y_test,y_pred_best)
mae_best=mean_absolute_error(y_test,y_pred_best)

print(f"Tuned Accuracy (R2): {r2_best*100:.2f}%")
print(f"Tuned Error (MAE):   {mae_best:.2f}")

plt.figure(figsize=(10,5))
importances=best_model.feature_importances_
sns.barplot(x=features,y=importances)
plt.title("Which Pollutant Affects AQI the Most?")
plt.ylabel("Importance Score")
save_plot("pollutant_effect.png")


plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_best,alpha=0.5,color='cyan',edgecolor='k')
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',lw=3)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Truth vs Prediction")
save_plot("Truth_Vs_Predict.png")

artifacts={
    'model':best_model,
    'encoder':le
}

joblib.dump(artifacts,"aqi_model.pkl")
print("Model and encoder successfully saved to 'aqi_model.pkl'")