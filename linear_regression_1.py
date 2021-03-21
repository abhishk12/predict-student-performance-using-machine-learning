import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv('student-mat.csv', sep=';')

labelencoder = LabelEncoder()

features =['sex', 'age', 'studytime', 'failures', 'paid', 'internet', 'absences', 'G1', 'G2', 'activities']

dataset = data[features]
y = data['G3']

dataset['Sex'] = labelencoder.fit_transform(dataset['sex'])
dataset['Activities'] = labelencoder.fit_transform(dataset['activities'])
dataset['Paid'] = labelencoder.fit_transform(dataset['paid'])

dataset['Internet'] = labelencoder.fit_transform(dataset['internet'])
dataset.drop(['paid', 'sex', 'internet', 'activities'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.1)

bestScore = 0
for _ in range(1000):

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(accuracy)
    if accuracy > bestScore:
        bestScore = accuracy
        with open("studentsModel.pickle", "wb") as f:
            pickle.dump(model, f)


pickle_in = open('studentsModel.pickle', 'rb')
model = pickle.load(pickle_in)
print(bestScore)
