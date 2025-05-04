from ortools.sat.python import cp_model

def main() -> None:

    model = cp_model.CpModel()

    var_upper_bound = max(47, 50, 37)
    x = model.new_int_var(0, var_upper_bound, "x")
    y = model.new_int_var(0, var_upper_bound, "y")
    z = model.new_int_var(0, var_upper_bound, "z")

    model.add(2*x + 7*y + 3*z <= 50)
    model.add(3*x - 5*y + 7*z <= 45)

    model.maximize(2*x + 2*y + 3*z)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Maximum of objective function: {solver.objective_value}\n")
        print(f"x = {solver.value(x)}")
        print(f"y = {solver.value(y)}")
        print(f"z = {solver.value(z)}")
    else:
        print("No solution found")

    print("Statistics:")
    print(f"status: {solver.status_name(status)}")
    print(f"conflicts: {solver.num_conflicts}")
    print(f"branches: {solver.num_branches}")
    print(f"wall time: {solver.wall_time}")

main()