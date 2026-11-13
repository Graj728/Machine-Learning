import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#loading the data set

df=pd.read_csv("Student_performance_data _.csv")
#Creation of x and y fro train and test split 

x=df.drop(columns=["StudentID","GPA","GradeClass"])
y=df["GradeClass"]

#splitting the data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

#creating the first model
model1=LogisticRegression(max_iter=1000,random_state=41)
model1.fit(x_train,y_train)

#Prediction making and calculating the accuarcy
predictions1=model1.predict(x_test)
accuracy1=accuracy_score(y_test,predictions1)
print("------------------------Model1 Summary----------------------- \n")

print("Accuracy: ",accuracy1)


#Creating the Second model

model2=SVC(random_state=41,max_iter=1000)
model2.fit(x_train,y_train)

#Prediction making and accuracy calculation for model2
predictions2=model2.predict(x_test)
accuracy2=accuracy_score(y_test,predictions2)

print("------------------------Model2 Summary----------------------- \n")

print("Accuracy: ",accuracy2)

#compzrison
#Logistic Regression performed better than SVM for this dataset. Logistic Regression achieved an accuracy of 70.56%, while SVM achieved an accuracy of 67.85%. 
# #Therefore, Logistic Regression was the better-performing model for this dataset.