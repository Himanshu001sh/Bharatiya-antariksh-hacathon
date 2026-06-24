import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

df = pd.read_csv("Chandigarh_Final_ML_Dataset_V2.csv")

X = df[
    [
        "Longitude",
        "Latitude",
        "NDVI",
        "NDBI",
        "Distance_to_Water_Meters",
        "Has_Building",
        "Has_Road"
    ]
]

y = df["LST_Celsius"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("XGBoost R²:", r2_score(y_test, pred))