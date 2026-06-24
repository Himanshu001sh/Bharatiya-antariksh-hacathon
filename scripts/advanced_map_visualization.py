import pandas as pd
import folium
from folium.plugins import HeatMap

df = pd.read_csv("Final_UHI_Results.csv")

# Create base map centered on Chandigarh
m = folium.Map(
    location=[30.7333, 76.7794],
    zoom_start=12,
    tiles="OpenStreetMap"
)

# -------------------------------
# 1. HEATMAP LAYER
# -------------------------------

heat_data = df[
    ["Latitude", "Longitude", "Predicted_LST"]
].dropna().values.tolist()

HeatMap(
    heat_data,
    radius=15,
    blur=20,
    max_zoom=13
).add_to(m)


# -------------------------------
# 2. RISK POINT LAYER
# -------------------------------

def get_color(risk):
    if risk == "High":
        return "red"
    elif risk == "Medium":
        return "orange"
    else:
        return "green"


# sample points to keep map fast
sample_df = df.sample(5000, random_state=42)

risk_layer = folium.FeatureGroup(name="Heat Risk Points")

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color=get_color(row["Heat_Risk"]),
        fill=True,
        fill_color=get_color(row["Heat_Risk"]),
        fill_opacity=0.7,
        popup=f"""
        <b>Predicted LST:</b> {row['Predicted_LST']:.2f} °C<br>
        <b>Risk:</b> {row['Heat_Risk']}<br>
        <b>NDVI:</b> {row['NDVI']:.3f}<br>
        <b>NDBI:</b> {row['NDBI']:.3f}<br>
        <b>Distance to Water:</b> {row['Distance_to_Water_Meters']:.2f} m<br>
        <b>Recommendation:</b> {row['Recommendation']}
        """
    ).add_to(risk_layer)

risk_layer.add_to(m)


# -------------------------------
# 3. TOP 100 HOTSPOTS
# -------------------------------

hotspot_layer = folium.FeatureGroup(name="Top 100 Hotspots")

top_hotspots = df.nlargest(100, "Predicted_LST")

for _, row in top_hotspots.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color="darkred",
        fill=True,
        fill_color="red",
        fill_opacity=0.9,
        popup=f"""
        <b>TOP HOTSPOT</b><br>
        <b>Predicted LST:</b> {row['Predicted_LST']:.2f} °C<br>
        <b>Risk:</b> {row['Heat_Risk']}<br>
        <b>Recommendation:</b> {row['Recommendation']}
        """
    ).add_to(hotspot_layer)

hotspot_layer.add_to(m)


# -------------------------------
# 4. LEGEND
# -------------------------------

legend_html = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 210px;
height: 145px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding: 10px;
">
<b>Heat Risk Legend</b><br>
<span style="color:green;">●</span> Low Risk<br>
<span style="color:orange;">●</span> Medium Risk<br>
<span style="color:red;">●</span> High Risk<br>
<span style="color:darkred;">●</span> Top Hotspots<br><br>
HeatMap = Predicted LST
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))


# -------------------------------
# 5. LAYER CONTROL
# -------------------------------

folium.LayerControl().add_to(m)


# -------------------------------
# 6. SAVE MAP
# -------------------------------

m.save("advanced_uhi_heatmap.html")

print("Advanced UHI map saved as advanced_uhi_heatmap.html")