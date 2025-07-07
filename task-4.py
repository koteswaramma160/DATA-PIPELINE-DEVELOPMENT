###### TASK-4 OPTIMIZATION MODEL

from pulp import *

# Define Linear Programming(LP) Problem
model = LpProblem("Maximize_Profit", LpMaximize)

# Decision Variables
x = LpVariable('Product_A', lowBound=0, cat='Continuous')
y = LpVariable('Product_B', lowBound=0, cat='Continuous')

# Objective Function
model += 20 * x + 30 * y, "Profit"

# Constraints
model += 2 * x + 1 * y <= 100  # Resource 1
model += 1 * x + 2 * y <= 80   # Resource 2

# Solve
model.solve()
print("Status:", LpStatus[model.status])
print(f"Produce {x.varValue} units of Product A")
print(f"Produce {y.varValue} units of Product B")
print("Max Profit: ₹", value(model.objective))

