import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def save(name):
    plt.savefig(name)
    plt.show()

df=pd.read_csv("Mall_Customers.csv")

df=df.rename(columns={
    'Annual Income (k$)': "income",
    "Spending Score (1-100)" : "score"
})

X=df[['income','score']]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#WCSS -(Within Cluster Sum of Sqaures)
wcss=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,init='k-means++',random_state=42,n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss,marker="*")
plt.title("Elbow Method")
plt.xlabel("no of Elements")
plt.ylabel("WCSS")
save("elbow_method_png")

#As per the elbow method we can take 5 
model=KMeans(n_clusters=5,random_state=42,n_init=10)
model.fit(X_scaled)

df['cluster']=model.labels_
plt.figure(figsize=(10,6))
sns.scatterplot(x="income",y='score',hue='cluster',data=df,palette="magma",s=100)
plt.title('Cluster of Customers')
plt.xlabel("Annual income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
save("Cluster.png")

