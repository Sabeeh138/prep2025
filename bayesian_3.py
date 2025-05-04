from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork(
    [
        ("Disease", "Fever"),
        ("Disease", "Cough"),
        ("Disease", "Fatigue"),
        ("Disease", "Chills")
    ]
)

cpd_disease = TabularCPD(
    variable="Disease",
    variable_card=2,
    values=[
        [0.3],
        [0.7]
    ],
    state_names={"Disease": ["Flu", "Cold"]}
)

cpd_fever = TabularCPD(
    variable="Fever",
    variable_card=2,
    values=[
        [0.9, 0.5],
        [0.1, 0.5]
    ],
    evidence=["Disease"],
    evidence_card=[2],
    state_names={
            'Fever': ['Yes', 'No'],
            'Disease': ["Flu", "Cold"]
    }
)

cpd_cough = TabularCPD(
    variable= "Cough",
    variable_card=2,
    values=[
        [0.8, 0.6],
        [0.2, 0.4]
    ],
    evidence=["Disease"],
    evidence_card=[2],
    state_names={
        "Disease":["Flu", "Cold"],
        "Cough": ["Yes", "No"]
    }
)
cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.7, 0.3],  
        [0.3, 0.7]   
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Fatigue': ['Yes', 'No'],
        'Disease': ['Flu', 'Cold']
    }
)

cpd_chills = TabularCPD(
    variable='Chills',
    variable_card=2,
    values=[
        [0.6, 0.4],  
        [0.4, 0.6]   
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Chills': ['Yes', 'No'],
        'Disease': ['Flu', 'Cold']
    }
)

model.add_cpds(cpd_chills, cpd_cough, cpd_disease, cpd_fatigue, cpd_fever)

assert model.check_model(), "Model is incorrect"

infer = VariableElimination(model)

print("\nTask 1: P(Disease | Fever=Yes, Cough=Yes)")
result1 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes'}
)
print(result1)

print("\nTask 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)")
result2 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes', 'Chills': 'Yes'}
)
print(result2)

print("\nTask 3: P(Fatigue = Yes | Disease = Flu)")
result3 = infer.query(
    variables=['Fatigue'],
    evidence={'Disease': 'Flu'}
)
print(result3)