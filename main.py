import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Reading the .csv File
train_data = pd.read_csv("dataset/train.csv")

#Data Collection & Processing
train_data.drop("Cabin", axis = 1, inplace = True)
train_data["Age"].fillna(train_data["Age"].mean(), inplace = True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace = True)

"""
#Visualizing the Data

#Count plot for the column "Survived"
sns.countplot(x = "Survived", data = train_data)
plt.plot()

#Count plot for the column "Sex"
sns.countplot(x = "Sex", data = train_data)
plt.plot()

#Count plot for the column "Survived", gender-wise
sns.countplot(x = "Sex", data = train_data, hue = "Survived")
plt.plot()

#Count plot for the column "Pclass", based on who Survived
sns.countplot(x = "Pclass", hue = "Survived", data = train_data)
plt.plot()
"""

#Categorical Encoding (All text values to computer understandable Numeric Values)

#male = 0, female = 1; s = 0, c = 1, q = 2
train_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace = True)

#Separating Features & Target
drop_col = train_data.drop(columns = ["PassengerId", "Name", "Ticket", "Survived"], axis = 1)
survived = train_data["Survived"]

#Splitting Training and Test Data
drop_col_train, drop_col_test, survived_train, survived_test = train_test_split(drop_col, survived, test_size = 0.2, random_state = 2)

#Model Training

#Logistic Regression
#X = Input Features; Y = Prediction Probability

X = drop_col
Y = survived
X_train, X_test = drop_col_train, drop_col_test
Y_train, Y_test = survived_train, survived_test

model = LogisticRegression()

#training the LogisticRegression model with the training data
model.fit(X_train, Y_train)

#the training model's accuracy
X_train_prediction = model.predict(X_train)
accuracy = accuracy_score(Y_train, X_train_prediction)

#the training model's accuracy for test values
X_test_prediction = model.predict(X_test)
accuracy_test = accuracy_score(Y_test, X_test_prediction)

#Printing the final accuracy scores
print("\n\n\n")
print(f"Accuracy Score of the Training Data is {accuracy*100}%")
print(f"Accuracy Score of the Test Data is {accuracy_test*100}%")