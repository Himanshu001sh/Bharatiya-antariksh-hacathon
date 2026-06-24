import numpy as np
import pandas as pd

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

df = pd.read_csv("Final_UHI_Results.csv")

# Take top 1 hotspot for demo
hotspot = df.sort_values("Predicted_LST", ascending=False).iloc[0]

current_temp = hotspot["Predicted_LST"]

# Decision variables:
# x[0] = tree units
# x[1] = cool roof units
# x[2] = water intervention units

class UHIOptimization(Problem):

    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=0,
            xl=np.array([0, 0, 0]),
            xu=np.array([50, 50, 20])
        )

    def _evaluate(self, X, out, *args, **kwargs):

        trees = X[:, 0]
        cool_roofs = X[:, 1]
        water_units = X[:, 2]

        # Assumed temperature reduction model
        temp_reduction = (
            0.5 * trees +
            0.3 * cool_roofs +
            0.8 * water_units
        )

        final_temp = current_temp - temp_reduction

        # Cost model
        cost = (
            500 * trees +
            300 * cool_roofs +
            2500 * water_units
        )

        # Objective 1: minimize final temperature
        # Objective 2: minimize cost
        out["F"] = np.column_stack([final_temp, cost])


problem = UHIOptimization()

algorithm = NSGA2(
    pop_size=100
)

res = minimize(
    problem,
    algorithm,
    ("n_gen", 100),
    seed=42,
    verbose=False
)

solutions = pd.DataFrame(
    res.X,
    columns=["Tree_Units", "Cool_Roof_Units", "Water_Units"]
)

solutions["Final_Temperature"] = res.F[:, 0]
solutions["Total_Cost"] = res.F[:, 1]
solutions["Temperature_Reduction"] = current_temp - solutions["Final_Temperature"]

solutions = solutions.sort_values("Total_Cost")

solutions.to_csv("NSGA2_Optimized_Solutions.csv", index=False)

print("Current Hotspot Temperature:", current_temp)
print(solutions.head(10))
print("Saved: NSGA2_Optimized_Solutions.csv")