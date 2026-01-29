import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 

# 1. Create Fake "Hiring" Data
# [Experience, Test Score, Interview Score]
X_train = np.array([
    [0, 8, 9], [1, 5, 6], [2, 6, 5], [3, 9, 9], [3, 4, 4], # Juniors
    [5, 7, 8], [6, 9, 7], [7, 3, 5], [8, 9, 9], [10, 5, 5], # Mid/Seniors
    [10, 10, 10], [0, 2, 2], [5, 5, 5], [2, 9, 9], [1, 1, 1]
])

# 0 = Rejected, 1 = Hired
y_train = np.array([
    0, 0, 0, 1, 0, # Note: High scores needed for juniors
    1, 1, 0, 1, 0, # Seniors need decency
    1, 0, 0, 1, 0
])
model=RandomForestClassifier(n_estimators=20,random_state=42)
model.fit(X_train,y_train)
try:
    exp=int(input("Enter the Experience of person : "))
    t_score=int(input("Enter the score in test : "))
    i_score=int(input("Enter the score in Interview : "))
except ValueError:
    print("Type correct values")
predict=model.predict([[exp,t_score,i_score]])
result="Hired" if predict[0]==1 else "Rejected"

importances = model.feature_importances_
feature_names = ['Experience', 'Written Test', 'Interview Score']

# Plotting
plt.bar(feature_names, importances, color=['blue', 'green', 'orange'])
plt.title("What Matters Most for Getting Hired?")
plt.ylabel("Importance Score (0-1)")
plt.savefig("example.png")
plt.show()