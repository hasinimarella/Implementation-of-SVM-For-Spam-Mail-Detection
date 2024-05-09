# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start the program.
2. Preprocessing the data.
3. Feature Extraction need to be done.
4. Training the SVM model that required for the given program.
5. Model Evalutaion is very essential process.,
6. Thus the program executed successfully.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MARELLA HASINI
RegisterNumber:212223240083
*/
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/spam.csv",encoding = 'windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x = data['v2'].values
y = data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 48)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
![alt text](<ml 9.png>)
![alt text](<ml 9.1.png>)
![alt text](<ml 9.2.png>)
![alt text](<Screenshot 2024-05-09 235045.png>)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
