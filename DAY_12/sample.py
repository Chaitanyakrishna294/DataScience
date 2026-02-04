import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

# 1. LOAD DATA
df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 2. TRAIN MODEL 🧠
# n_clusters=5 (We confirmed this from the Dendrogram/Elbow yesterday)
# linkage='ward' (Minimizes variance, makes tight clusters)
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)

# 3. VISUALIZE GROUPS 🎨
plt.figure(figsize=(10, 6))

# Plot the points, colored by their cluster (y_hc)
sns.scatterplot(
    x=X[:, 0], 
    y=X[:, 1], 
    hue=y_hc, 
    palette='bright', 
    s=100, 
    edgecolor='black'
)

plt.title('Clusters of Customers (Hierarchical)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# 4. ANALYSIS 📝
df['Cluster'] = y_hc
print("\n--- Cluster Averages ---")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())