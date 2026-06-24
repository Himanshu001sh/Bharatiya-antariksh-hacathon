import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Final_UHI_Results.csv")

# GRAPH: Heat risk distribution
risk_counts = df["Heat_Risk"].value_counts()

plt.figure(figsize=(7, 5))
risk_counts.plot(kind="bar")
plt.title("Heat Risk Distribution Across Chandigarh")
plt.xlabel("Heat Risk Level")
plt.ylabel("Number of Locations")
plt.tight_layout()
plt.savefig("heat_risk_distribution.png")
plt.show()


# GRAPH: NDVI vs Temperature
plt.figure(figsize=(7, 5))
plt.scatter(df["NDVI"], df["Predicted_LST"], alpha=0.3)
plt.title("Vegetation Impact on Surface Temperature")
plt.xlabel("NDVI")
plt.ylabel("Predicted LST (°C)")
plt.tight_layout()
plt.savefig("ndvi_vs_temperature.png")
plt.show()


# GRAPH: Distance to Water vs Temperature
plt.figure(figsize=(7, 5))
plt.scatter(df["Distance_to_Water_Meters"], df["Predicted_LST"], alpha=0.3)
plt.title("Distance from Water Bodies vs Surface Temperature")
plt.xlabel("Distance to Water (Meters)")
plt.ylabel("Predicted LST (°C)")
plt.tight_layout()
plt.savefig("water_distance_vs_temperature.png")
plt.show()


# GRAPH: Before vs After Mitigation
current_avg = df["Predicted_LST"].mean()

# simple assumption for presentation:
# vegetation + cool roof interventions reduce average temperature by 3°C
after_mitigation = current_avg - 3

plt.figure(figsize=(6, 5))
plt.bar(["Current", "After Mitigation"], [current_avg, after_mitigation])
plt.title("Expected Temperature Reduction After Mitigation")
plt.ylabel("Average Surface Temperature (°C)")
plt.tight_layout()
plt.savefig("before_after_mitigation.png")
plt.show()

print("Health impact graphs saved successfully!")