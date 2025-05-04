from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

cpd_burglary = TabularCPD(
    variable='Burglary',
    variable_card=2,
    values=[[0.999], [0.001]]
)

cpd_earthquake = TabularCPD(
    variable='Earthquake',
    values=[[0.998], 
            [0.002]],
    variable_card=2,
)

cpd_alarm = TabularCPD(
    variable='Alarm',
    variable_card= 2,
    values=[[0.99, 0.71, 0.06, 0.05],
            [0.001, 0.29, 0.94, 0.95]],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

cpd_johnCalls = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values = [
        [0.3, 0.9],
        [0.7, 0.1]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

cpd_mary = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.2, 0.99],
        [0.8, 0.01]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

model.add_cpds(cpd_burglary, cpd_earthquake, cpd_johnCalls, cpd_mary, cpd_alarm)

assert model.check_model(), "Model is incorrect"

inference = VariableElimination(model=model)

result = inference.query(variables = ['Burglary'], evidence = {'JohnCalls': 1, 'MaryCalls': 1})
print(result)