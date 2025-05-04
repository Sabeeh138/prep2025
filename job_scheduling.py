from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Job durations and number of jobs
job_durations = [3, 2, 2]
num_jobs = len(job_durations)
horizon = sum(job_durations)  # Maximum total time needed

# Create start time variables for each job
start_0 = model.new_int_var(0, horizon, "start_0")
start_1 = model.new_int_var(0, horizon, "start_1")
start_2 = model.new_int_var(0, horizon, "start_2")

# Create interval variables (for no-overlap constraint)
interval_0 = model.new_interval_var(start_0, job_durations[0], start_0 + job_durations[0], "interval_0")
interval_1 = model.new_interval_var(start_1, job_durations[1], start_1 + job_durations[1], "interval_1")
interval_2 = model.new_interval_var(start_2, job_durations[2], start_2 + job_durations[2], "interval_2")

# Ensure no jobs overlap
model.add_no_overlap([interval_0, interval_1, interval_2])

# Optional: Minimize makespan (when the last job finishes)
makespan = model.new_int_var(0, horizon, "makespan")
model.add(makespan >= start_0 + job_durations[0])
model.add(makespan >= start_1 + job_durations[1])
model.add(makespan >= start_2 + job_durations[2])
model.minimize(makespan)

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output the result
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Job Schedule:")
    print(f"Job 1 starts at time {solver.value(start_0)}")
    print(f"Job 2 starts at time {solver.value(start_1)}")
    print(f"Job 3 starts at time {solver.value(start_2)}")
    print(f"Total time (makespan): {solver.value(makespan)}")
else:
    print("No solution found.")
