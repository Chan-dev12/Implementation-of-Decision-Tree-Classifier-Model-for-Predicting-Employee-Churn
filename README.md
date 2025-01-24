# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the employee churn dataset (handle missing values, encode categorical data).
2.Split the dataset into training and testing sets.
3.Initialize the DecisionTreeClassifier model with appropriate hyperparameters.
4.Train the model using the training dataset.
5.Evaluate the model on the test dataset and predict employee churn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: chanthru v
RegisterNumber: 24900997
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
data["left"].value_counts()
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
# print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_sprmd_company","work_accident","promotion_last_5years","salary"]]
# print(x.head()) 
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
# print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```


## Output:
![ex 8(1)](https://github.com/user-attachments/assets/89c31021-889e-4f03-9982-458ed6d980a3)
![exp8(2)](https://github.com/user-attachments/assets/4bacb0b9-b86e-4348-90df-c8597e722c87)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
