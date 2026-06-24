import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# I Use sample only because SHAP can be slow
X_sample = X_test.sample(100, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Summary 
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

# barplot
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP images saved")