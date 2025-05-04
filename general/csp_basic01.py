from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_values = 3
x = model.new_int_var(0, num_values-1, "x")
y = model.new_int_var(0, num_values-1, "y")
z = model.new_int_var(0, num_values-1, "z")

model.add(x != y)

solver = cp_model.CpSolver()
status = solver.solve(model)

if status == cp_model.OPTIMAL or cp_model.FEASIBLE:
    print(f"x = {solver.value(x)}")
    print(f"y = {solver.value(y)}")
    print(f"z = {solver.value(z)}")

else:
    print("No solution found")

    