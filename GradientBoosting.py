import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib  


file_path = "internships_india.csv"  
df = pd.read_csv(file_path)


features = ["City", "Sector", "Duration_Months", "Remote"]
target = "Stipend"


encoder_city = LabelEncoder()
encoder_sector = LabelEncoder()
df["City"] = encoder_city.fit_transform(df["City"])
df["Sector"] = encoder_sector.fit_transform(df["Sector"])
df["Remote"] = df["Remote"].map({"Yes": 1, "No": 0})


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)


y_pred = gb_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")


joblib.dump(gb_model, "stipend_prediction_model.joblib")


joblib.dump(encoder_city, "encoder_city.joblib")
joblib.dump(encoder_sector, "encoder_sector.joblib")

print("Model and encoders saved successfully!")
