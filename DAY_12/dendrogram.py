import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

df=pd.read_csv("Mall_Customers.csv")
df=df.rename(columns={"Annual Income (k$)":"income","Spending Score (1-100)":"score"})
X=df[["income","score"]].values

plt.figure(figsize=(10,10))
dendeogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.savefig("dendrogram.png")
plt.show()