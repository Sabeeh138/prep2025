from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Regions and colors
regions = ['A', 'B', 'C', 'D']
num_colors = 3  # Red, Green, Blue
color_names = ['Red', 'Green', 'Blue']

# Explicitly create variables for each region
A = model.new_int_var(0, num_colors - 1, "A")
B = model.new_int_var(0, num_colors - 1, "B")
C = model.new_int_var(0, num_colors - 1, "C")
D = model.new_int_var(0, num_colors - 1, "D")

# Adjacency constraints (no two adjacent regions can have the same color)
model.add(A != B)  # A cannot be the same color as B
model.add(A != C)  # A cannot be the same color as C
model.add(B != C)  # B cannot be the same color as C
model.add(C != D)  # C cannot be the same color as D

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output the result
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Map Coloring Solution:")
    # Print each region's color explicitly
    color_index_A = solver.value(A)
    print(f"Region A: {color_names[color_index_A]}")
    
    color_index_B = solver.value(B)
    print(f"Region B: {color_names[color_index_B]}")
    
    color_index_C = solver.value(C)
    print(f"Region C: {color_names[color_index_C]}")
    
    color_index_D = solver.value(D)
    print(f"Region D: {color_names[color_index_D]}")
else:
    print("No solution found.")

