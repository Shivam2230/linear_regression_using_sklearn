#import required libraries...
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#read csv file using pandas
dataset=pd.read_csv('/Weather.csv')

#use only first 1000 rows
dataset=dataset.head(10000)

#print dataset shape
print(dataset.shape)

#plot scatter plot using pandas plot() function
dataset.plot(x='MinTemp',y='MaxTemp',style='o')

#reg will be our linear regression model
reg=LinearRegression()

#reshape X and Y to (-1,1) where -1 denotes that it can be of any dimension but col will be 1 only
X=dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

#split dataset into train and test sets 20% data will be test set 80% will be train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fit train data into model
reg.fit(X_train,y_train)

#predict from model
y_pred=reg.predict(X_test)

slope=reg.coef_
print(slope)

intercept=reg.intercept_
print(intercept)

#plot test set data and predicted data using a red line
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.show()
