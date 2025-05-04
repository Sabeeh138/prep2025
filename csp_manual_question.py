from ortools.sat.python import cp_model

frequencies = [15, 8, 20]
volumes = [2, 1, 3]
distances = [1, 2, 3]
slot_capacities = [3, 3, 3]

num_products = len(frequencies)
num_slots = len(distances)

model = cp_model.CpModel()

# Flattened list
assign = [
    model.new_bool_var(f"assign_p{p}_s{s}")
    for p in range(num_products)
    for s in range(num_slots)
]

# Constraint: each product in exactly one slot
for p in range(num_products):
    model.add(sum(assign[p * num_slots + s] for s in range(num_slots)) == 1)

# Constraint: slot capacities are not exceeded
for s in range(num_slots):
    model.add(
        sum(assign[p * num_slots + s] * volumes[p] for p in range(num_products))
        <= slot_capacities[s]
    )

# Objective: minimize walking cost
model.minimize(
    sum(assign[p * num_slots + s] * frequencies[p] * distances[s]
        for p in range(num_products) for s in range(num_slots))
)

# Solve
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Assignment:")
    for p in range(num_products):
        for s in range(num_slots):
            if solver.value(assign[p * num_slots + s]):
                print(f"Product {p+1} assigned to Slot {s+1}")
    print(f"Total Walking Cost: {solver.objective_value}")
else:
    print("No feasible solution found.")
