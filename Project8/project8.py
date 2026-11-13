# import of required libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#task1:description of data and dropping columns for processing the data
df=pd.read_csv("healthcare_dataset.csv")
print(df.shape)        #Shape of dataFRame
print(df.head())  # first five rows of the dataset
print(df.columns)  #colms in the dataset
print(df.describe()) # some descibed calculations for dataset
print("missing values per column:",df.isnull().sum())
print("Dupilcate rows:",df.duplicated().sum())
#Dropping the duplicate so we dont use same record for training and testing
## I have decided to drop the date of admission as well as date of discharge as I think they do not generalize new patients and calendar dates are not a meaningful numeric feature
df=df.drop_duplicates().reset_index(drop=True)
print("Shape after dropping duplicates:",df.shape)


y=df["Test Results"]
x=df.drop(columns=["Test Results","Name", "Doctor","Hospital", "Room Number","Date of Admission", "Discharge Date"])
category_columns=["Gender", "Blood Type", "Medical Condition", "Insurance Provider", "Admission Type", "Medication"]
number_columns=["Age","Billing Amount"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=42)
### Encoder help was implemented with help from chatgpt
encoded=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
encoded.fit(x_train[category_columns])
names=encoded.get_feature_names_out(category_columns)
enc_train=pd.DataFrame(encoded.transform(x_train[category_columns]),columns=names,index=x_train.index)
enc_test=pd.DataFrame(encoded.transform(x_test[category_columns]),columns=names,index=x_test.index)
x_train=pd.concat([x_train[number_columns],enc_train],axis=1)
x_test=pd.concat([x_test[number_columns],enc_test],axis=1)

print("Features after encoding",x_train.shape[1])

#Task2: Feature Selection(train data)
#method1:using mutual info
discrete=~x_train.columns.isin(number_columns)
mut_in_score=mutual_info_classif(x_train,y_train,discrete_features=discrete,random_state=42)
mut_in=pd.Series(mut_in_score,index=x_train.columns)

#method 2: using Random forest classifier
forest=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
forest.fit(x_train,y_train)
rf=pd.Series(forest.feature_importances_,index=x_train.columns)
scores=pd.DataFrame({"mutual info":mut_in,"random forest importance":rf})
scores=scores.sort_values("random forest importance",ascending=False)
#keeping a feature if it is above the average in score
keepm=(scores["mutual info"]>mut_in.mean())|(scores["random forest importance"]>rf.mean())
scores["decision"]=keepm.map({True:"keep",False:"remove"})
print(scores)
keep=scores.index[keepm]
print("keep",list(keep))
scores["random forest importance"].head(15).sort_values().plot.barh(title="Top 15 features of Random Forest Importance")
plt.tight_layout()
plt.show()

#model1
model1=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
model1.fit(x_train,y_train)
prediction1=model1.predict(x_test)
accuracy_all=accuracy_score(y_test,prediction1)
#model2
model2=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
model2.fit(x_train[keep],y_train)
prediction2=model2.predict(x_test[keep])
accuracy_reduced=accuracy_score(y_test,prediction2)

#comparison of both models
print(f"Model1:{x_train.shape[1]} features,accuracy{accuracy_all:.4f}")
print(f"Model2:{len(keep)} features,accuracy{accuracy_reduced:.4f}")
