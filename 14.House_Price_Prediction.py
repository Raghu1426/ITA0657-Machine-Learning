import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
dataset = pd.read_csv("HousePricePrediction.csv")
print("Dataset Preview:")
print(dataset.head())
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
dataset = dataset.dropna()
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("\nCategorical variables:", len(object_cols))
num_cols = list(dataset.select_dtypes(include=['int64', 'float64']).columns)
num_cols.remove("SalePrice")
print("Numerical variables:", len(num_cols))
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)
X = df_final.drop(["SalePrice"], axis=1)
y = df_final["SalePrice"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, random_state=42)
model_SVR = SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_SVR = model_SVR.predict(X_valid)
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_LR = model_LR.predict(X_valid)
print("\n🔍 Model Performance:")
print(f"SVR - Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(Y_valid, Y_pred_SVR):.4f}")
print(f"SVR - R² Score: {r2_score(Y_valid, Y_pred_SVR):.4f}")
print(f"Linear Regression - Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(Y_valid, Y_pred_LR):.4f}")
print(f"Linear Regression - R² Score: {r2_score(Y_valid, Y_pred_LR):.4f}")
plt.figure(figsize=(10, 5))
plt.scatter(Y_valid, Y_pred_LR, alpha=0.7, color='blue', label="Linear Regression")
plt.scatter(Y_valid, Y_pred_SVR, alpha=0.7, color='green', label="SVR")
plt.plot([min(Y_valid), max(Y_valid)], [min(Y_valid), max(Y_valid)], linestyle="dashed", color="red")  # Perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
new_house = pd.DataFrame([[3, 2, 1500, 1, 2005, 5000]],
                          columns=num_cols[:6])
new_house_encoded = pd.DataFrame(OH_encoder.transform(pd.DataFrame([["Category1"]], columns=object_cols)))
new_house_encoded.columns = OH_encoder.get_feature_names_out()
new_house_final = pd.concat([new_house, new_house_encoded], axis=1)
new_house_scaled = scaler.transform(new_house_final)
predicted_price_LR = model_LR.predict(new_house_scaled)
predicted_price_SVR = model_SVR.predict(new_house_scaled)
print(f"\n🏠 Predicted Price for New House (Linear Regression): ${predicted_price_LR[0]:.2f}")
print(f"🏠 Predicted Price for New House (SVR): ${predicted_price_SVR[0]:.2f}")