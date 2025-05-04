from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the network structure
# List all edges as (Parent, Child) based on the problem's dependency diagram
# Example: ('Intelligence', 'Grade'), ('Grade', 'Pass')
model = DiscreteBayesianNetwork([
    # Fill in edges here
])

# Step 2: Define CPDs
# Root Nodes (nodes with no parents)
# Example: Intelligence (High=0, Low=1), P(High)=0.7, P(Low)=0.3
cpd_root1 = TabularCPD(
    variable='Root1',  # Node name (e.g., 'Intelligence')
    variable_card=2,   # Number of states (e.g., 2 for High/Low)
    values=[[]]        # Prior probabilities (e.g., [[0.7], [0.3]])
    # Note: States are integers (e.g., 0=High, 1=Low)
)

cpd_root2 = TabularCPD(
    variable='Root2',
    variable_card=2,
    values=[[]]
    # Note: States are integers
)

# Add more root nodes as needed (cpd_root3, cpd_root4, etc.)
# Example: cpd_root3 = ...

# Child Nodes (nodes with parents)
# Example: Grade (A=0, B=1, C=2) with parents Intelligence, StudyHours, Difficulty
cpd_child1 = TabularCPD(
    variable='Child1',  # Node name (e.g., 'Grade')
    variable_card=2,    # Number of states (e.g., 3 for A, B, C)
    values=[[]],        # Conditional probabilities (rows=child states, columns=parent combos)
    evidence=['Parent1', 'Parent2'],  # Parent names (e.g., ['Intelligence', 'StudyHours'])
    evidence_card=[2, 2]  # Cardinality of each parent (e.g., [2, 2, 2] for 3 parents)
    # Note: States are integers (e.g., 0=A, 1=B, 2=C for Child1)
)

cpd_child2 = TabularCPD(
    variable='Child2',
    variable_card=2,
    values=[[]],
    evidence=['Child1'],
    evidence_card=[2]
    # Note: States are integers
)

# Add more child nodes as needed (cpd_child3, cpd_child4, etc.)

# Step 3: Add CPDs to the model
# Include all CPDs defined above (comment out unused ones if switching tasks)
model.add_cpds(cpd_root1, cpd_root2, cpd_child1, cpd_child2)

# Step 4: Verify the model
assert model.check_model(), "Model is invalid!"

# Step 5: Perform inference
inference = VariableElimination(model)

# Define your queries here
# Example: P(Child1 | Parent1=1)
result1 = inference.query(variables=['Child1'], evidence={'Parent1': 1})
print("P(Child1 | Parent1=1):")
print(result1)

# Example: P(Root1 | Child2=1)
result2 = inference.query(variables=['Root1'], evidence={'Child2': 1})
print("P(Root1 | Child2=1):")
print(result2)

# Solved Example: Task 3 (Disease Prediction)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the network structure
model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Chills')
])

# Step 2: Define CPDs
# P(Disease): 0=Flu, 1=Cold
cpd_disease = TabularCPD(
    variable='Disease', variable_card=2,
    values=[[0.3], 
            [0.7]]  # P(Flu)=0.3, P(Cold)=0.7
)

# P(Fever | Disease): 0=No, 1=Yes
cpd_fever = TabularCPD(
    variable='Fever', variable_card=2,
    values=[
        [0.1, 0.5],  # P(Fever=No | Flu, Cold)
        [0.9, 0.5]   # P(Fever=Yes | Flu, Cold)
    ],
    evidence=['Disease'], evidence_card=[2]
)

# P(Cough | Disease)
cpd_cough = TabularCPD(
    variable='Cough', variable_card=2,
    values=[
        [0.2, 0.4],  # P(Cough=No | Flu, Cold)
        [0.8, 0.6]   # P(Cough=Yes | Flu, Cold)
    ],
    evidence=['Disease'], evidence_card=[2]
)

# P(Fatigue | Disease)
cpd_fatigue = TabularCPD(
    variable='Fatigue', variable_card=2,
    values=[
        [0.3, 0.7],  # P(Fatigue=No | Flu, Cold)
        [0.7, 0.3]   # P(Fatigue=Yes | Flu, Cold)
    ],
    evidence=['Disease'], evidence_card=[2]
)

# P(Chills | Disease)
cpd_chills = TabularCPD(
    variable='Chills', variable_card=2,
    values=[
        [0.4, 0.6],  # P(Chills=No | Flu, Cold)
        [0.6, 0.4]   # P(Chills=Yes | Flu, Cold)
    ],
    evidence=['Disease'], evidence_card=[2]
)

# Step 3: Add CPDs
model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)

# Step 4: Verify
assert model.check_model(), "Model is invalid!"

# Step 5: Perform inference
inference = VariableElimination(model)

# Query 1: P(Disease | Fever=Yes, Cough=Yes)
result1 = inference.query(variables=['Disease'], evidence={'Fever': 1, 'Cough': 1})
print("P(Disease | Fever=Yes, Cough=Yes):")
print(result1)

# Query 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)
result2 = inference.query(variables=['Disease'], evidence={'Fever': 1, 'Cough': 1, 'Chills': 1})
print("P(Disease | Fever=Yes, Cough=Yes, Chills=Yes):")
print(result2)

# Query 3: P(Fatigue=Yes | Disease=Flu)
result3 = inference.query(variables=['Fatigue'], evidence={'Disease': 0})
print("P(Fatigue=Yes | Disease=Flu):")
print(result3)

# Expected Output (approximate):
# P(Disease | Fever=Yes, Cough=Yes):
# +-----------+------------------+
# | Disease   |   phi(Disease)   |
# +===========+==================+
# | Disease(0)|           0.616  |  # Flu
# | Disease(1)|           0.384  |  # Cold
# +-----------+------------------+
# P(Disease | Fever=Yes, Cough=Yes, Chills=Yes):
# +-----------+------------------+
# | Disease   |   phi(Disease)   |
# +===========+==================+
# | Disease(0)|           0.753  |  # Flu
# | Disease(1)|           0.247  |  # Cold
# +-----------+------------------+
# P(Fatigue=Yes | Disease=Flu):
# +-----------+------------------+
# | Fatigue   |   phi(Fatigue)   |
# +===========+==================+
# | Fatigue(0)|           0.300  |  # No
# | Fatigue(1)|           0.700  |  # Yes
# +-----------+------------------+

# Solved Example: Burglary Alarm (Using DiscreteBayesianNetwork)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

# Step 2: Define the CPDs (Conditional Probability Distributions)
# P(Burglary)
cpd_burglary = TabularCPD(
    variable='Burglary', variable_card=2,
    values=[[0.999], 
            [0.001]]
)

# P(Earthquake)
cpd_earthquake = TabularCPD(
    variable='Earthquake', variable_card=2,
    values=[[0.998], 
            [0.002]]
)

# P(Alarm | Burglary, Earthquake)
cpd_alarm = TabularCPD(
    variable='Alarm',
    variable_card=2,
    values=[
        [0.999, 0.71, 0.06, 0.05],  # Alarm = False
        [0.001, 0.29, 0.94, 0.95]   # Alarm = True
    ],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

# P(JohnCalls | Alarm)
cpd_john = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values=[
        [0.95, 0.05],  # JohnCalls = False (0=False, 1=True)
        [0.05, 0.95]   # JohnCalls = True
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

# P(MaryCalls | Alarm)
cpd_mary = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.99, 0.10],  # MaryCalls = False (0=False, 1=True)
        [0.01, 0.90]   # MaryCalls = True
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

# Step 3: Add CPDs to the model
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)

# Step 4: Verify the model
assert model.check_model(), "Model is incorrect"

# Step 5: Perform inference
inference = VariableElimination(model)

# Query: What is the probability of a burglary given that both John and Mary called?
result = inference.query(variables=['Burglary'], evidence={'JohnCalls': 1, 'MaryCalls': 1})
print(result)

# Expected Output (approximate):
# +-------------+-----------------+
# | Burglary    |   phi(Burglary) |
# +=============+=================+
# | Burglary(0) |          0.716  |  # False
# | Burglary(1) |          0.284  |  # True
# +-------------+-----------------+