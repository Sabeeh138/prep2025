from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

x = df[['age', 'fare']].fillna(df[['age', 'fare']].mean())
y = df['survived']

x= pd.DataFrame(x)
y= pd.Series(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()
accuracy_scores = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)


print("K-Fold CV Average Accuracy: ", np.mean(accuracy_scores))