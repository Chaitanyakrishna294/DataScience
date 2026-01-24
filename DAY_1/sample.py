from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# Height,weight,shoe_size
x=[[180,80,44],[190,90,43],[160,60,38],[170,70,40],[175,75,42],[185,85,41],[155,50,36],[165,65,39],[200,100,45],[150,45,35]]
y=['male','male','female','female','male','male','female','female','male','female']
clf1=tree.DecisionTreeClassifier()
clf2=KNeighborsClassifier(n_neighbors=3)
clf3=svm.SVC()
clf4=RandomForestClassifier(n_estimators=10)
clf5=GaussianNB()
clf1=clf1.fit(x,y)
clf2=clf2.fit(x,y)
clf3=clf3.fit(x,y)
clf4=clf4.fit(x,y)
clf5=clf5.fit(x,y)
Height=int(input("Enter the height:"))
Weight=int(input("Enter the weight:"))
Shoe_size=int(input("Enter the shoe size:"))
prediction1=clf1.predict([[Height,Weight,Shoe_size]])
prediction2=clf2.predict([[Height,Weight,Shoe_size]])
prediction3=clf3.predict([[Height,Weight,Shoe_size]])
prediction4=clf4.predict([[Height,Weight,Shoe_size]])
prediction5=clf5.predict([[Height,Weight,Shoe_size]])
print(f"Decision Tree Classifier: {prediction1}")
print(f"KNN Classifier: {prediction2}")
print(f"SVM :{prediction3}")
print(f"Random Forest Classifier: {prediction4}")
print(f"Naive Bayes Classifier: {prediction5}")
