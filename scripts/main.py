
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
#%%
df = pd.read_csv("Chandigarh_Final_ML_Dataset_V2.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

#%%

X = df[
    [    "Longitude",
        "Latitude",
        "NDVI",
        "NDBI",
        "Distance_to_Water_Meters",
        "Has_Building",
        "Has_Road"
    ]
]

y = df["LST_Celsius"]
#%%

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#%%

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
#%%

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
#%%

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print(importance)
#%%

df["Predicted_LST"] = model.predict(X)


def classify_risk(temp):
    if temp < 38:
        return "Low"
    elif temp < 42:
        return "Medium"
    else:
        return "High"

df["Heat_Risk"] = df["Predicted_LST"].apply(classify_risk)

#%%
def recommend_solution(row):
    solutions = []

    if row["NDVI"] < 0.2:
        solutions.append("Increase tree cover / green spaces")

    if row["NDBI"] > 0.1:
        solutions.append("Use cool roofs and reflective surfaces")

    if row["Distance_to_Water_Meters"] > 1000:
        solutions.append("Create water bodies / cooling zones")

    if row["Has_Road"] == 1:
        solutions.append("Add roadside plantation")

    if row["Has_Building"] == 1:
        solutions.append("Promote rooftop gardens")

    if len(solutions) == 0:
        return "No urgent action needed"

    return "; ".join(solutions)

df["Recommendation"] = df.apply(recommend_solution, axis=1)

#%%
df["Priority_Score"] = (
    df["Predicted_LST"] * 0.5
    + df["Distance_to_Water_Meters"] * 0.0005
    - df["NDVI"] * 10
)

#%%

print(df[[
    "Longitude",
    "Latitude",
    "LST_Celsius",
    "Predicted_LST",
    "Heat_Risk",
    "Recommendation",
    "Priority_Score"
]].head())

#%%
df.to_csv("Final_UHI_Results.csv", index=False)

print("Final_UHI_Results.csv saved ")