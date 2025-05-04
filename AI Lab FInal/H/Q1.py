'''Question 1: Network Failure Diagnosis
Problem Statement:
A network administrator is diagnosing failures in a computer network. The network’s performance depends on three factors: ServerStatus (Up/Down), RouterStatus (Functional/Faulty), and Bandwidth (High/Low). 
These factors influence NetworkPerformance (Good/Poor), which in turn affects UserComplaints (Yes/No). Construct a Bayesian Network to model this scenario, define the CPDs, and answer the following queries:

What is the probability of Poor NetworkPerformance given RouterStatus=Faulty and Bandwidth=Low?
What is the probability of ServerStatus=Up given UserComplaints=Yes and Bandwidth=High?

Network Structure:

Nodes: ServerStatus, RouterStatus, Bandwidth, NetworkPerformance, UserComplaints.
Dependencies:
ServerStatus → NetworkPerformance
RouterStatus → NetworkPerformance
Bandwidth → NetworkPerformance
NetworkPerformance → UserComplaints
Provided CPDs:

P(ServerStatus): P(Up)=0.9, P(Down)=0.1
P(RouterStatus): P(Functional)=0.85, P(Faulty)=0.15
P(Bandwidth): P(High)=0.7, P(Low)=0.3
P(UserComplaints | NetworkPerformance):
P(Yes | Good)=0.1, P(No | Good)=0.9
P(Yes | Poor)=0.8, P(No | Poor)=0.2
Task:

Assume a reasonable CPD for P(NetworkPerformance | ServerStatus, RouterStatus, Bandwidth).
Implement the Bayesian Network using pgmpy with DiscreteBayesianNetwork.
Compute the two inference queries.
Difficulty:

Multi-parent node (NetworkPerformance with three parents, like Grade in Task 1).
Requires assuming a complex CPD (8 parent combinations).
Reverse inference query (P(ServerStatus | UserComplaints, Bandwidth)).'''

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the network structure
# List all edges as (Parent, Child) based on the problem's dependency diagram
model = DiscreteBayesianNetwork([
    ('ServerStatus', 'NetworkPerformance'),
    ('RouterStatus', 'NetworkPerformance'),
    ('Bandwidth', 'NetworkPerformance'),
    ('NetworkPerformance', 'UserComplaints')
])

# Step 2: Define CPDs
# Root Nodes (nodes with no parents)
# ServerStatus (Up=0, Down=1), P(Up)=0.9, P(Down)=0.1
cpd_ServerStatus = TabularCPD(
    variable='ServerStatus', variable_card=2,
    values=[[0.9], 
            [0.1]]  # P(Up)=0.9, P(Down)=0.1
)

# RouterStatus (Functional=0, Faulty=1), P(Functional)=0.85, P(Faulty)=0.15
cpd_RouterStatus = TabularCPD(
    variable='RouterStatus', variable_card=2,
    values=[
        [0.85],  
        [0.15]   
    ],
)

# Bandwidth (High=0, Low=1), P(High)=0.7, P(Low)=0.3
cpd_Bandwidth = TabularCPD(
    variable='Bandwidth', variable_card=2,
    values=[
        [0.7],  
        [0.3]   
    ],
)

# Child Nodes (nodes with parents)
# NetworkPerformance (Good=0, Poor=1) with parents ServerStatus, RouterStatus, Bandwidth
cpd_NetworkPerformance = TabularCPD(
    variable='NetworkPerformance', variable_card=2,
    values=[
        [0.9, 0.8, 0.6, 0.4, 0.8, 0.6, 0.3, 0.1],  # P(Good | Up/Func/High, Up/Func/Low, ...)
        [0.1, 0.2, 0.4, 0.6, 0.2, 0.4, 0.7, 0.9]   # P(Poor | ...)
    ],
    evidence=['ServerStatus', 'RouterStatus', 'Bandwidth'], evidence_card=[2, 2, 2]
)

# UserComplaints (Yes=0, No=1) with parent NetworkPerformance
cpd_UserComplaints = TabularCPD(
    variable='UserComplaints', variable_card=2,
    values=[
        [0.1, 0.8],  # P(Yes | Good, Poor)
        [0.9, 0.2]   # P(No | Good, Poor)
    ],
    evidence=['NetworkPerformance'], evidence_card=[2]
)

# Step 3: Add CPDs
# Include all CPDs defined above
model.add_cpds(cpd_ServerStatus, cpd_RouterStatus, cpd_Bandwidth, cpd_NetworkPerformance, cpd_UserComplaints)

# Step 4: Verify
assert model.check_model(), "Model is invalid!"

# Step 5: Perform inference
inference = VariableElimination(model)

# Define your queries here
# Query 1: P(NetworkPerformance | RouterStatus=Faulty, Bandwidth=Low)
result1 = inference.query(variables=['NetworkPerformance'], evidence={'RouterStatus': 1, 'Bandwidth': 1})
print("P(NetworkPerformance | RouterStatus=Faulty, Bandwidth=Low):")
print(result1)

# Query 2: P(ServerStatus | UserComplaints=Yes, Bandwidth=High)
result2 = inference.query(variables=['ServerStatus'], evidence={'UserComplaints': 0, 'Bandwidth': 0})
print("P(ServerStatus | UserComplaints=Yes, Bandwidth=High):")
print(result2)