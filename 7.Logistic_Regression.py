import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1)
lr_model = LinearRegression()
lr_model.fit(X, y)
print('Coefficients: ', lr_model.coef_)
print('Intercept: ', lr_model.intercept_)
plt.scatter(X, y, color='blue')
plt.plot(X, lr_model.predict(X), color='red', linewidth=3)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
