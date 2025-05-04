# CSP Examples: Job Scheduling and Map Coloring

# This file demonstrates CSP solutions using the python-constraint library.
# It includes examples for job scheduling and map coloring with flexible constraint options.
# Install the library if needed: pip install python-constraint

from constraint import Problem

# --- Job Scheduling Example ---
def job_scheduling_example():
    print("\n--- Job Scheduling CSP ---")
    
    # Define the variables (start times for each job)
    variables = ['T1', 'T2', 'T3', 'T4', 'T5']  # T1 is start time of J1, T2 for J2, etc.
    
    # Define the durations for each job
    durations = {'T1': 3, 'T2': 5, 'T3': 6, 'T4': 3, 'T5': 4}
    
    # Define the time horizon (total available time)
    horizon = 21  # Increased to 21 minutes to accommodate total duration
    
    # Define domains: possible start times for each job
    # Ti can start from 0 to (horizon - duration_i), adjusted to avoid overlap
    domains = {
        'T1': [0],  # J1 must start at time 0
        'T2': list(range(horizon - durations['T2'] + 1)),  # 0 to 16
        'T3': list(range(horizon - durations['T3'] + 1)),  # 0 to 15
        'T4': list(range(horizon - durations['T2'] + 1)),  # 0 to 16 (after T2)
        'T5': list(range(horizon - durations['T5'] + 1))   # 0 to 17
    }
    
    # Create the problem
    problem = Problem()
    
    # Add variables with their domains
    for var in variables:
        problem.addVariable(var, domains[var])
    
    # --- Active Constraints ---

    # J3 before J2: T3 + 6 <= T2
    problem.addConstraint(lambda t3, t2: t3 + 6 <= t2, ['T3', 'T2'])
    
    # J2 and J4 can’t be parallel: T4 must start after J2 ends (T2 + 5 <= T4)
    problem.addConstraint(lambda t2, t4: t2 + 5 <= t4, ['T2', 'T4'])
    
    # J5 last: T5 starts after each other job finishes
    problem.addConstraint(lambda t5: t5 >= 3, ['T5'])  # After J1
    problem.addConstraint(lambda t5, t2: t5 >= t2 + 5, ['T5', 'T2'])  # After J2
    problem.addConstraint(lambda t5, t3: t5 >= t3 + 6, ['T5', 'T3'])  # After J3
    problem.addConstraint(lambda t5, t4: t5 >= t4 + 3, ['T5', 'T4'])  # After J4
    
    # --- Additional Constraints (Commented Out) ---
    # Uncomment the constraints required by your exam question

    # 1. J1 and J3 must be in sequence (J1 before J3)
    # problem.addConstraint(lambda t1, t3: t1 + 3 <= t3, ['T1', 'T3'])

    # 2. J2 must start after a minimum time (e.g., 2 minutes)
    # problem.addConstraint(lambda t2: t2 >= 2, ['T2'])

    # 3. J4 and J5 must overlap partially
    # problem.addConstraint(lambda t4, t5: t4 < t5 + 4 and t5 < t4 + 3, ['T4', 'T5'])

    # 4. J2 and J3 are on the same machine and can’t overlap
    # problem.addConstraint(lambda t2, t3: (t2 + 5 <= t3) or (t3 + 6 <= t2), ['T2', 'T3'])

    # 5. J2 must start exactly 2 minutes after J1
    # problem.addConstraint(lambda t1, t2: t2 == t1 + 2, ['T1', 'T2'])

    # 6. J1 and J5 must be on different machines (e.g., simulated by non-overlap with a large gap)
    # problem.addConstraint(lambda t1, t5: t1 + 3 + 5 <= t5 or t5 + 4 + 5 <= t1, ['T1', 'T5'])

    # 7. J3 has a deadline of 10 minutes
    # problem.addConstraint(lambda t3: t3 + 6 <= 10, ['T3'])

    # 8. J4 must start before a maximum time (e.g., 12 minutes)
    # problem.addConstraint(lambda t4: t4 <= 12, ['T4'])

    # 9. J1 and J2 must be separated by at least 4 minutes
    # problem.addConstraint(lambda t1, t2: t2 >= t1 + 3 + 4 or t1 >= t2 + 5 + 4, ['T1', 'T2'])

    # 10. J5 must finish by a specific deadline (e.g., 20 minutes)
    # problem.addConstraint(lambda t5: t5 + 4 <= 20, ['T5'])

    # 11. J2 and J5 cannot overlap on the same resource
    # problem.addConstraint(lambda t2, t5: t2 + 5 <= t5 or t5 + 4 <= t2, ['T2', 'T5'])

    # 12. J3 must start after J1 finishes with a delay of 2 minutes
    # problem.addConstraint(lambda t1, t3: t3 >= t1 + 3 + 2, ['T1', 'T3'])

    # 13. J4 has a maximum processing time limit (e.g., must finish by 15 minutes)
    # problem.addConstraint(lambda t4: t4 + 3 <= 15, ['T4'])

    # 14. J1 and J4 must be scheduled on consecutive time slots (e.g., within 3 minutes)
    # problem.addConstraint(lambda t1, t4: abs(t4 - t1) <= 3, ['T1', 'T4'])

    # 15. J2 cannot start before 5 minutes
    # problem.addConstraint(lambda t2: t2 >= 5, ['T2'])

    # 16. J3 and J5 must be at least 3 minutes apart
    # problem.addConstraint(lambda t3, t5: t5 >= t3 + 6 + 3 or t3 >= t5 + 4 + 3, ['T3', 'T5'])

    # 17. J1 must be the only job at time 0
    # problem.addConstraint(lambda t2, t3, t4, t5: t2 > 0 and t3 > 0 and t4 > 0 and t5 > 0, ['T2', 'T3', 'T4', 'T5'])

    # 18. J4 and J2 must have a minimum gap of 2 minutes
    # problem.addConstraint(lambda t2, t4: t4 >= t2 + 5 + 2 or t2 >= t4 + 3 + 2, ['T2', 'T4'])

    # 19. J5 has a preferred start time range (e.g., 15-18 minutes)
    # problem.addConstraint(lambda t5: 15 <= t5 <= 18, ['T5'])

    # 20. J1 and J3 must not overlap on a shared resource
    # problem.addConstraint(lambda t1, t3: t1 + 3 <= t3 or t3 + 6 <= t1, ['T1', 'T3'])

    # 21. J2 must finish before a deadline of 13 minutes
    # problem.addConstraint(lambda t2: t2 + 5 <= 13, ['T2'])

    # 22. J4 must start after J3 ends
    # problem.addConstraint(lambda t3, t4: t3 + 6 <= t4, ['T3', 'T4'])

    # 23. J1 and J5 must be scheduled with a maximum gap of 5 minutes
    # problem.addConstraint(lambda t1, t5: abs(t5 - t1) <= 5, ['T1', 'T5'])

    # 24. J3 cannot be scheduled in the first 2 minutes
    # problem.addConstraint(lambda t3: t3 >= 2, ['T3'])

    # 25. J2 and J5 must be on different machines (non-overlap with gap)
    # problem.addConstraint(lambda t2, t5: t2 + 5 + 5 <= t5 or t5 + 4 + 5 <= t2, ['T2', 'T5'])

    # 26. J4 must be completed within 10 minutes of J1’s start
    # problem.addConstraint(lambda t1, t4: t4 + 3 <= t1 + 10, ['T1', 'T4'])

    # 27. J1 and J2 must overlap partially if on the same machine
    # problem.addConstraint(lambda t1, t2: t1 < t2 + 5 and t2 < t1 + 3, ['T1', 'T2'])

    # 28. J5 must start after a global minimum time (e.g., 10 minutes)
    # problem.addConstraint(lambda t5: t5 >= 10, ['T5'])

    # 29. J3 and J4 must be separated by at least 1 minute
    # problem.addConstraint(lambda t3, t4: t4 >= t3 + 6 + 1 or t3 >= t4 + 3 + 1, ['T3', 'T4'])

    # 30. J2 has a preferred end time (e.g., before 12 minutes)
    # problem.addConstraint(lambda t2: t2 + 5 <= 12, ['T2'])

    # Solve the problem
    solution = problem.getSolution()
    
    if solution:
        print("Solution found:", solution)
        # Verify solution
        t1, t2, t3, t4, t5 = solution['T1'], solution['T2'], solution['T3'], solution['T4'], solution['T5']
        print(f"J1: {t1} to {t1 + durations['T1']}")
        print(f"J2: {t2} to {t2 + durations['T2']}")
        print(f"J3: {t3} to {t3 + durations['T3']}")
        print(f"J4: {t4} to {t4 + durations['T4']}")
        print(f"J5: {t5} to {t5 + durations['T5']}")
    else:
        print("No solution exists within 21 minutes.")
# --- Map Coloring Example ---
def map_coloring_example():
    print("\n--- Map Coloring CSP ---")
    
    # Define the variables (regions to color)
    variables = ['WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T']
    
    # Define the domains (possible colors)
    colors = ['Red', 'Green', 'Blue']
    domains = {var: colors for var in variables}
    
    # Create the problem
    problem = Problem()
    
    # Add variables with their domains
    for var in variables:
        problem.addVariable(var, domains[var])
    
    # Define constraints: adjacent regions must have different colors
    adjacencies = [
        ('WA', 'NT'), ('WA', 'SA'),
        ('NT', 'SA'), ('NT', 'Q'),
        ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
        ('Q', 'NSW'),
        ('NSW', 'V')
    ]
    for pair in adjacencies:
        problem.addConstraint(lambda a, b: a != b, pair)
    
    # Additional constraint examples (uncomment or modify)
    # 1. If WA must be Red:
    # domains['WA'] = ['Red']
    
    # 2. If SA and V must have the same color:
    # problem.addConstraint(lambda sa, v: sa == v, ['SA', 'V'])
    
    # 3. If Q can’t be Blue:
    # domains['Q'] = [color for color in domains['Q'] if color != 'Blue']
    
    # Solve the problem
    solution = problem.getSolution()
    
    if solution:
        print("Solution found:", solution)
    else:
        print("No solution exists.")

# --- Sudoku Example ---
def sudoku_example():
    print("\n--- Sudoku CSP ---")
    
    # Define a 4x4 Sudoku grid (simpler for demonstration; 9x9 works similarly)
    # Variables are cell positions (row, col), e.g., '00' for row 0, col 0
    variables = [f"{r}{c}" for r in range(4) for c in range(4)]
    
    # Define domains: numbers 1 to 4 (for 4x4 Sudoku)
    domains = {var: list(range(1, 5)) for var in variables}
    
    # Pre-filled cells (example partial Sudoku grid)
    # 0 0 1 0
    # 0 4 0 0
    # 0 0 2 0
    # 0 3 0 0
    pre_filled = {
        '02': 1,  # Row 0, Col 2 = 1
        '11': 4,  # Row 1, Col 1 = 4
        '22': 2,  # Row 2, Col 2 = 2
        '31': 3   # Row 3, Col 1 = 3
    }
    for var, value in pre_filled.items():
        domains[var] = [value]
    
    # Create the problem
    problem = Problem()
    
    # Add variables with their domains
    for var in variables:
        problem.addVariable(var, domains[var])
    
    # Define constraints: uniqueness in rows, columns, and 2x2 subgrids
    # 1. Row constraints: each number appears once per row
    for r in range(4):
        row_vars = [f"{r}{c}" for c in range(4)]
        problem.addConstraint(lambda *args: len(set(args)) == len(args), row_vars)
    
    # 2. Column constraints: each number appears once per column
    for c in range(4):
        col_vars = [f"{r}{c}" for r in range(4)]
        problem.addConstraint(lambda *args: len(set(args)) == len(args), col_vars)
    
    # 3. Subgrid constraints: each number appears once per 2x2 subgrid
    for sr in range(0, 4, 2):  # Subgrid rows: 0, 2
        for sc in range(0, 4, 2):  # Subgrid cols: 0, 2
            subgrid_vars = [f"{r}{c}" for r in range(sr, sr + 2) for c in range(sc, sc + 2)]
            problem.addConstraint(lambda *args: len(set(args)) == len(args), subgrid_vars)
    
    # Additional constraint examples (uncomment or modify)
    # 1. If cell (0,0) must be 2:
    # domains['00'] = [2]
    
    # 2. If row 0 must sum to a specific value (e.g., 10):
    # problem.addConstraint(lambda a, b, c, d: a + b + c + d == 10, ['00', '01', '02', '03'])
    
    # 3. If cell (1,1) can’t be 3:
    # domains['11'] = [val for val in domains['11'] if val != 3]
    
    # Solve the problem
    solution = problem.getSolution()
    
    if solution:
        print("Solution found:")
        # Print the Sudoku grid
        for r in range(4):
            row = [solution[f"{r}{c}"] for c in range(4)]
            print(row)
    else:
        print("No solution exists.")
# --- Timetabling Example ---
def timetabling_example():
    print("\n--- Timetabling CSP ---")
    
    # Define courses, teachers, days, and slots
courses = ['Math', 'Physics', 'AI']
teachers = ['ProfA', 'ProfB', 'ProfC']
days = ['Mon', 'Tue', 'Wed']
slots = ['9AM', '10AM', '11AM']

# Pre-assigned course-teacher mappings
course_teacher_map = {
    'Math': 'ProfA',
    'Physics': 'ProfB',
    'AI': 'ProfC'
}

# Generate domains: (teacher, day, slot) tuples, restricted to the assigned teacher
domain = [(t, d, s) for t in teachers for d in days for s in slots]
domains = {course: [tpl for tpl in domain if tpl[0] == course_teacher_map[course]] for course in courses}

# Initialize the CSP problem
problem = Problem()

# Add variables and their domains
for course in courses:
    problem.addVariable(course, domains[course])

# --- Active Constraints ---

# 1. Room availability: Only one course per (day, slot) - assumes one room
for i, course1 in enumerate(courses):
    for course2 in courses[i+1:]:
        problem.addConstraint(
            lambda a, b: (a[1], a[2]) != (b[1], b[2]),  # Compare (day, slot)
            (course1, course2)
        )

# 2. Teacher preference: ProfA cannot teach on Wednesdays
for course in courses:
    if course_teacher_map[course] == 'ProfA':
        problem.addConstraint(
            lambda a: a[1] != 'Wed',
            (course,)
        )

# --- Additional Constraints (Commented Out) ---
# Uncomment the constraints required by your exam question

# 3. Student conflicts: Math and AI cannot be scheduled at the same time
# problem.addConstraint(
#     lambda a, b: (a[1], a[2]) != (b[1], b[2]),
#     ('Math', 'AI')
# )

# 4. Teacher workload: ProfB cannot teach more than one class per day
# for day in days:
#     profb_courses = [c for c in courses if course_teacher_map[c] == 'ProfB']
#     if len(profb_courses) > 1:
#         problem.addConstraint(
#             lambda *args: sum(1 for a in args if a[1] == day) <= 1,
#             profb_courses
#         )

# 5. Course dependency: Physics must be scheduled after Math
# problem.addConstraint(
#     lambda a, b: days.index(a[1]) < days.index(b[1]) or (a[1] == b[1] and slots.index(a[2]) < slots.index(b[2])),
#     ('Math', 'Physics')
# )

# 6. Time preference: AI must be at 9AM
# problem.addConstraint(
#     lambda a: a[2] == '9AM',
#     ('AI',)
# )

# 7. Lunch break: No classes at 12PM (requires adding '12PM' to slots if used)
# for course in courses:
#     problem.addConstraint(
#         lambda a: a[2] != '12PM',
#         (course,)
#     )

# 8. Day conflict: Math and Physics cannot be on the same day
# problem.addConstraint(
#     lambda a, b: a[1] != b[1],
#     ('Math', 'Physics')
# )

# 9. Teacher availability: ProfC is only available on Tue and Wed
# for course in courses:
#     if course_teacher_map[course] == 'ProfC':
#         problem.addConstraint(
#             lambda a: a[1] in ['Tue', 'Wed'],
#             (course,)
#         )

# 10. Slot preference: Math must be in the morning (before 11AM)
# problem.addConstraint(
#     lambda a: a[2] in ['9AM', '10AM'],
#     ('Math',)
# )

# 11. Teacher day preference: ProfB prefers Mon or Tue
# for course in courses:
#     if course_teacher_map[course] == 'ProfB':
#         problem.addConstraint(
#             lambda a: a[1] in ['Mon', 'Tue'],
#             (course,)
#         )

# 12. Consecutive classes: Math and AI must be on consecutive days
# problem.addConstraint(
#     lambda a, b: abs(days.index(a[1]) - days.index(b[1])) == 1,
#     ('Math', 'AI')
# )

# 13. No early classes: ProfA refuses 9AM slots
# for course in courses:
#     if course_teacher_map[course] == 'ProfA':
#         problem.addConstraint(
#             lambda a: a[2] != '9AM',
#             (course,)
#         )

# 14. Course spacing: Physics and AI must be at least one day apart
# problem.addConstraint(
#     lambda a, b: abs(days.index(a[1]) - days.index(b[1])) >= 1,
#     ('Physics', 'AI')
# )

# 15. Slot exclusion: No 11AM slots for ProfC
# for course in courses:
#     if course_teacher_map[course] == 'ProfC':
#         problem.addConstraint(
#             lambda a: a[2] != '11AM',
#             (course,)
#         )

# 16. Day exclusion: No classes on Mon for AI
# problem.addConstraint(
#     lambda a: a[1] != 'Mon',
#     ('AI',)
# )

# 17. Teacher slot limit: ProfA can only teach one slot across all days
# profa_courses = [c for c in courses if course_teacher_map[c] == 'ProfA']
# if len(profa_courses) > 1:
#     problem.addConstraint(
#         lambda *args: len(set((a[1], a[2]) for a in args)) == len(args),
#         profa_courses
#     )

# 18. Cross-teacher conflict: ProfA and ProfB cannot teach at the same time
# profa_courses = [c for c in courses if course_teacher_map[c] == 'ProfA']
# profb_courses = [c for c in courses if course_teacher_map[c] == 'ProfB']
# for c1 in profa_courses:
#     for c2 in profb_courses:
#         problem.addConstraint(
#             lambda a, b: (a[1], a[2]) != (b[1], b[2]),
#             (c1, c2)
#         )

# --- Solve and Display Solution ---
solution = problem.getSolution()

if solution:
    print("Solution found:", solution)
    print("\nTimetable:")
    for course, (teacher, day, slot) in solution.items():
        print(f"{course} taught by {teacher} on {day} at {slot}")
else:
    print("No solution exists.")

# --- Main Execution ---
if __name__ == "__main__":
    job_scheduling_example()
    map_coloring_example()
    sudoku_example()
    timetabling_example()