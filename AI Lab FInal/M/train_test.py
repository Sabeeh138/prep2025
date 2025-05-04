from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

x = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy: .2f}")