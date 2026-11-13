import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#part 1: Load and explore the data

df=pd.read_csv("employee_data.csv")   # load the dataset into a pandas DataFrame
print(df.head())       # first five rows
print(df.shape)        # number of rows and columns
print(df.columns)      # column names
print(df.describe())   # basic descriptive statistics


#part 2: Categorizing the salary

# Converting a numeric salary into a Low / Medium / High category
def categorize_salary(salary):
    if salary<100000:
        return "Low"
    if salary<150000:
            return "Medium"
    else:
         return "High"

# Applying the function to every salary to create the new target column
df["Salary_Category"]=df["Salary"].apply(categorize_salary)
counts=df["Salary_Category"].value_counts()   # number of employees in each category
print(counts)

# Visualization: bar chart of the salary category distribution
order=["Low","Medium","High"]   # fixed display order for the bars
counts.reindex(order).plot(kind="bar",color=["blue","green","red"])
plt.title("Distribution of Salary")
plt.xlabel("Category of Salary")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)   # keep the labels horizontal
plt.tight_layout()       # prevent labels from being cut off
plt.show()

#part 3: Prepare the data

# Remove ID (just an identifier), Salary (the target was built from it),
# and Salary_Category (the target itself) so only the predictors remain
x=df.drop(columns=["ID","Salary","Salary_Category"])
# Encoding the text columns so the models can work with numbers
x=pd.get_dummies(x,columns=["Gender","Position"])
y=df["Salary_Category"]   # target variable the models will predict


#part 4: Splitting into training and testing data

# 80% of the data trains the models, 20% is held back to test them on unseen data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Train size: ",x_train.shape[0],"Test size: ",x_test.shape[0])

#part 5: Decision Tree model
dtc_model=DecisionTreeClassifier(random_state=42)   # create the model
dtc_model.fit(x_train,y_train)                      # train on the training data
dtc_prediction=dtc_model.predict(x_test)            # predict categories for the test data

#part 6: Random Forest model
rcf_model=RandomForestClassifier(n_estimators=100,random_state=42)   # forest of 10 trees
rcf_model.fit(x_train,y_train)                      # train on the training data
rcf_prediction=rcf_model.predict(x_test)            # predict categories for the test data

#part 7: Evaluating both models using accuracy
# Accuracy = fraction of test predictions that match the true categories
tree_accuracy=accuracy_score(y_test,dtc_prediction)
rcf_accuracy=accuracy_score(y_test,rcf_prediction)
print("Decision tree accuracy: ",tree_accuracy)
print("Random Forest Accuracy: ",rcf_accuracy)

#part 8: Feature importance from the Random Forest
# Pairing each feature name with its importance score, then sort from highest to lowest
importance=pd.Series(rcf_model.feature_importances_,index=x.columns)
importance=importance.sort_values(ascending=False)
print(importance)

# Bar graph of the 10 most important features
importance.head(10).plot(kind="bar")
plt.title("Most Important Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()