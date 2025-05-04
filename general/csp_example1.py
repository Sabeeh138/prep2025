from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_vals = 3
x = model.new_int_var(0, num_vals - 1, "x")
y = model.new_int_var(0, num_vals - 1, "y")
z = model.new_int_var(0, num_vals - 1, "z")
model.add(x != y)
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v} = {self.value(v)}", end = " ")
        print()

    @property
    def solution_count(self) -> int:
        return self.__solution_count
    
solver = cp_model.CpSolver()
solution_printer= VarArraySolutionPrinter([x, y, z])

solver.parameters.enumerate_all_solutions = True
status = solver.solve(model, solution_printer)
