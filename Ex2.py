import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data=pd.read_csv("car_data.csv")
x=data.get("width")
y=data.get("height")
print(x)
print(y)
plt.plot(x,y)
plt.title("Decision Graph")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train,y)
c=dtree.predict(x_test)
print(c)
comax=confusion_matrix(c,y_test)
corep=classification_report(c,y_test)
print(comax)
plt.show()
