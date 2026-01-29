import numpy as np
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression 
#X is No of hours studied
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
#Y is Scores
Y = np.array([15, 25, 35, 48, 55, 68, 72, 85, 90, 96])

model=LinearRegression() #select model
model.fit(X,Y) #train model
Hours=int(input("Enter the no of Hours He/She Studied:"))
predict=model.predict([[Hours]]) #predict 
print(f"Marks: {predict[0]:.2f}")

plt.scatter(X,Y,color='cyan',label="Actual Scores")
plt.plot(X,model.predict(X),color='red',label="Predicted Scores")
plt.xlabel('hours studied')
plt.ylabel('scores')
plt.legend()
plt.show()