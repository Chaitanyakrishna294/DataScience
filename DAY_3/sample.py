import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#X is the age of customer
X = np.array([22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49]).reshape(-1, 1)
#y is whether is bought(1) or not (0)
y = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
model=LogisticRegression()
model.fit(X,y)
Age=int(input("Enter the age:"))
predict=model.predict([[Age]])
print(f"if the age of the person is{Age} then it might be a chance of {predict} (1=bought,0=not)")

X_test=np.linspace(10,70,100).reshape(-1,1)
y_prob=model.predict_proba(X_test)[:,1]
plt.scatter(X,y,color='cyan',label='Data Points')
plt.plot(X_test,y_prob,color='red',label='Logestic Regression')
plt.title("Probability of Buying Insurence Vs Age")
plt.xlabel=('Age of Customers')
plt.ylabel=('probability(0=No,1=Yes)')
plt.legend()
plt.show()