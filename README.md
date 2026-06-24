# Bharatiya-antariksh-hacathon
## Urban Heat Island Detection & Mitigation using AI
# Overview :

This project uses satellite-derived environmental indicators and machine learning to detect Urban Heat Islands (UHI) in Chandigarh, explain the factors contributing to heat buildup, and recommend cost-effective mitigation strategies.

# Objectives :
Detect Urban Heat Island hotspots
Predict Land Surface Temperature (LST)
Explain model decisions using Explainable AI (SHAP)
Assess heat-related risk zones
Recommend mitigation strategies
Optimize interventions using NSGA-II

# Tech Stack :
     Data Processing
         Python
         Pandas
         NumPy
     Machine Learning
         Scikit-Learn
         Random Forest
         XGBoost
     Explainable AI
         SHAP
     Optimization
         NSGA-II (pymoo)
     Visualization
         Folium
        Matplotlib
        Streamlit
# Dataset Features:
     Feature	Description
        Longitude	Geographic coordinate
        Latitude	Geographic coordinate
        NDVI	Vegetation Index
        NDBI	Built-up Index
        Distance_to_Water_Meters	Distance from nearest water body
        Has_Building	Building presence
        Has_Road	Road presence
        LST_Celsius	Land Surface Temperature
#  Model Performance ::
        Model	R² Score
           Random Forest	0.956
           XGBoost	0.90

### Random Forest was selected as the final model.

# Explainable AI Insights

SHAP analysis revealed:

NDVI is the strongest environmental driver.
Water proximity significantly affects temperature.
Built-up areas contribute to heat accumulation.
Urban Heat Islands exhibit strong spatial patterns.
Generated Outputs

## Final_UHI_Results.csv
Contains:

Predicted LST
Heat Risk
Recommendations
Priority Score

## NSGA2_Optimized_Solutions.csv
Contains:

Intervention plans
Estimated temperature reduction
Estimated implementation cost
Dashboard Modules
UHI Heat Map
SHAP Explainability
Health Impact Analysis
Optimization Results



