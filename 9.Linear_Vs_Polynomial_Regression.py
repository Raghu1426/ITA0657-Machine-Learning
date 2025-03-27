import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
X_new = np.array([6]).reshape(-1, 1)
y_pred = reg.predict(X_new)
plt.scatter(X, y)
plt.plot(X, reg.predict(X), color='red')
plt.show()
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
reg = LinearRegression().fit(X_poly, y)
X_new = np.array([6]).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
y_pred = reg.predict(X_new_poly)
plt.scatter(X, y)
plt.plot(X, reg.predict(X_poly), color='red')
plt.show()
