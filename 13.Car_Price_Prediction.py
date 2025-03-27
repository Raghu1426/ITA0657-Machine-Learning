import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv("CarPrice.csv")
print("Dataset Columns:", list(data.columns))
data.columns = data.columns.str.strip()
predict = "price"
features = ["symboling", "wheelbase", "carlength", "carwidth", "carheight",
            "curbweight", "enginesize", "boreratio", "stroke", "compressionratio",
            "horsepower", "peakrpm", "citympg", "highwaympg"]
missing_features = [col for col in features if col not in data.columns]
if missing_features:
    print(f"Error: The following features are missing from the dataset: {missing_features}")
    exit()
X = data[features]
y = data[predict]
X = X.apply(pd.to_numeric, errors='coerce')
if X.isnull().sum().sum() > 0:
    print("Warning: NaN values detected in dataset. Replacing with column means.")
    X.fillna(X.mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="dashed", color="red")  # Perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
new_car = pd.DataFrame([[3, 100.4, 175.6, 66.5, 52.4, 2337, 109, 3.19, 3.40, 9.0, 102, 5500, 24, 30]],
                        columns=features)
predicted_price = model.predict(new_car)
print(f"\nPredicted Price for New Car: ${predicted_price[0]:.2f}")