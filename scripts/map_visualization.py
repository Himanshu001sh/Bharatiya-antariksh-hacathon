import pandas as pd
import folium
from folium.plugins import HeatMap
df = pd.read_csv("Final_UHI_Results.csv")

# Chandigarh center
m = folium.Map(
    location=[30.7333, 76.7794],
    zoom_start=12
)

def get_color(risk):
    if risk == "High":
        return "red"
    elif risk == "Medium":
        return "orange"
    else:
        return "green"

# use sample because 166k points can be heavy
sample_df = df.sample(5000, random_state=42)

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color=get_color(row["Heat_Risk"]),
        fill=True,
        fill_color=get_color(row["Heat_Risk"]),
        fill_opacity=0.7,
        popup=f"""
        Predicted LST: {row['Predicted_LST']:.2f} °C<br>
        Risk: {row['Heat_Risk']}<br>
        Recommendation: {row['Recommendation']}
        """
    ).add_to(m)

m.save("uhi_heat_risk_map.html")

print("Map saved as uhi_heat_risk_map.html")