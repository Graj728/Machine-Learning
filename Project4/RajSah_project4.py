import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

file=r"D:\Python\Machine learning\regression\Mobile-Price-Prediction-cleaned_data.csv"
df=pd.read_csv(file)

# print(df.columns)
# print(df.dtypes)

corr_coeff=df.corr(numeric_only=True)['Price'].drop('Price')
print(corr_coeff)
best_variable=corr_coeff.abs().idxmax()
best_corrcoef=corr_coeff[best_variable]
print("best variable =",best_variable)
print("Correlation Coefficient =",best_corrcoef)

x_sml=df[[best_variable]]
y=df['Price']

##Training the Simple model using Scikit-Learn
x_train_s,x_test_s,y_train_s,y_test_s=train_test_split(x_sml,y,test_size=0.2,random_state=42)
simple_model=LinearRegression()
simple_model.fit(x_train_s,y_train_s)
simple_prediction=simple_model.predict(x_test_s)
simple_mse=mean_squared_error(y_test_s,simple_prediction)
simple_r2=r2_score(y_test_s,simple_prediction)
print("Simple Linear Regression")
print("Intercept: ",simple_model.intercept_)
print("Coefficient: ",simple_model.coef_)
print("MSE: ",simple_mse)
print("R2: ",simple_r2)

#Predicting the price of phone based on simple model
simple_results = pd.DataFrame({
    "Actual Price": y_test_s.values,
    "Predicted Price": simple_prediction.round(0),
}).head(10)
print("Simple model - sample predictions:\n", simple_results)

##plotting the results of Simple Linear Regression

plt.figure(figsize=(8,5))
plt.scatter(x_test_s,y_test_s,alpha=0.7)
plt.plot(x_test_s,simple_prediction,color='blue')
plt.xlabel(best_variable)
plt.ylabel("Price")
plt.title("Simple linear regression: "+best_variable+"Price")
plt.show()


##Multiple Linear Regression
x_multiple=df.drop("Price",axis=1)
x_train_m,x_test_m,y_train_m,y_test_m=train_test_split(x_multiple,y,test_size=0.2,random_state=42)
multi_model=LinearRegression()
multi_model.fit(x_train_m,y_train_m)
multi_prediction=multi_model.predict(x_test_m)
multi_mse=mean_squared_error(y_test_m,multi_prediction)
multi_r2=r2_score(y_test_m,multi_prediction)
print("multiple Linear Regression")
print("Intercept: ",multi_model.intercept_)
print("Coefficient: ",multi_model.coef_)
print("MSE: ",multi_mse)
print("R2: ",multi_r2)

# Predictions: first 10 test phones, actual vs predicted
multi_results = pd.DataFrame({
    "Actual Price": y_test_m.values,
    "Predicted Price": multi_prediction.round(0),
}).head(10)
print("Multiple model - sample predictions:\n", multi_results)

##Plotting the results of multiple linear regression
plt.figure(figsize=(8,5))
plt.scatter(y_test_m,multi_prediction,alpha=0.6)
plt.plot(
    [y_test_m.min(),y_test_m.max()],
    [y_test_m.min(),y_test_m.max()],
    'r--'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple linear regression: Actual Price vs Predicted Price")
plt.show()



# ANALYSIS: ACCURACY AND RELIABILITY
# (values from running this script with random_state=42)
#
#   Simple (Ratings):  MSE ~ 272,877,387   R2 ~ 0.354
#   Multiple (all):    MSE ~ 239,357,657   R2 ~ 0.433
#
# Accuracy:
#   MSE is the average squared difference between actual and predicted price,
#   so lower is better. The multiple model has the lower MSE (about 12% lower)
#   and the higher R2, so it is more accurate: it uses RAM, ROM, battery and
#   the other features together instead of Ratings alone. Still, the square
#   root of MSE is about 15,500-16,500, so predictions are often off by
#   thousands of money. Neither model is highly accurate.
#
# Reliability:
#   R2 of 0.35 to 0.43 means more than half of the variation in Price is not
#   explained by either model. In the plots the fitted line misses many phones:
#   expensive phones are under-predicted and some cheap phones get negative
#   predicted prices. A straight line cannot follow this data well, so both
#   models are only moderately reliable.
#
# Conclusion:
#   Multiple linear regression is the better model, but neither is reliable
#   enough for exact price prediction. They are better for rough estimates.
