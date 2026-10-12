import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# House area (sq.ft) vs price (Rs. lakhs)
data = pd.DataFrame({
    "area":  [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000, 2200, 2400],
    "price": [32, 38, 45, 50, 58, 63, 72, 77, 86, 92, 101, 112],
})
X = data[["area"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Slope (m)      :", round(model.coef_[0], 4))
print("Intercept (c)  :", round(model.intercept_, 4))
print("MAE            :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE           :", round(mean_squared_error(y_test, pred) ** 0.5, 3))
print("R2 score       :", round(r2_score(y_test, pred), 4))
print("Price of 1800 sq.ft house (lakhs):",
      round(model.predict(pd.DataFrame({"area": [1800]}))[0], 2))

# Graphs: regression line and residual plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X_train, y_train, color="royalblue", label="Train data")
ax[0].scatter(X_test, y_test, color="orange", label="Test data")
ax[0].plot(X, model.predict(X), color="green", label="Fitted line")
ax[0].set_xlabel("Area (sq.ft)")
ax[0].set_ylabel("Price (lakhs)")
ax[0].set_title("Area vs Price")
ax[0].legend()

residual = y - model.predict(X)
ax[1].scatter(model.predict(X), residual, color="purple")
ax[1].axhline(0, color="black", linestyle="--")
ax[1].set_xlabel("Predicted price")
ax[1].set_ylabel("Residual (actual - predicted)")
ax[1].set_title("Residual plot")
plt.tight_layout()
plt.savefig("q2_regression.png", dpi=130)
plt.show()
