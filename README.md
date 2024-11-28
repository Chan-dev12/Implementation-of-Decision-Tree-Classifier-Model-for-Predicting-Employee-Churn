# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: chanthru v
RegisterNumber: 24900997 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv(r"C:\ml experinment\Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
data["left"].value_counts()
 
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
#  print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
#  print(x.head())    
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
#  print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

## Output:
![exp 8(1)](https://github.com/user-attachments/assets/5fb4c1f0-21f1-4ca0-8134-ea1eec8737aa)
![exp 8(2)](https://github.com/user-attachments/assets/c7bd7161-e7f0-4441-9a6b-50df2a7a24d1)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
