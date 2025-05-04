"""
New Question: Weather Impact on Outdoor Event Success
Problem Statement:
A city is planning an outdoor festival and wants to model the impact of weather conditions on the event’s success. 
The key factors are WeatherCondition (Sunny, Cloudy, Rainy), Temperature (Hot, Mild, Cold), and Wind (Low, High). These influence CrowdAttendance (High, Medium, Low), 
which affects EventSuccess (Successful, Moderate, Unsuccessful). Additionally, EventSuccess influences MediaCoverage (Positive, Neutral, Negative). 
Construct a Bayesian Network, define the CPDs, and answer the following queries:


What is the probability of Low CrowdAttendance given WeatherCondition=Rainy and Wind=High?
What is the probability of WeatherCondition=Sunny given EventSuccess=Successful?
What is the probability of MediaCoverage=Positive given CrowdAttendance=High and Temperature=Mild?
Network Structure:

Nodes: WeatherCondition, Temperature, Wind, CrowdAttendance, EventSuccess, MediaCoverage.
Dependencies:
WeatherCondition → CrowdAttendance
Temperature → CrowdAttendance
Wind → CrowdAttendance
CrowdAttendance → EventSuccess
EventSuccess → MediaCoverage
WeatherCondition → Temperature (to add a subchild dependency)
Provided CPDs:

P(WeatherCondition): P(Sunny)=0.5, P(Cloudy)=0.3, P(Rainy)=0.2
P(Wind): P(Low)=0.6, P(High)=0.4
P(Temperature): 0.4 0.4 0.2
P(EventSuccess | CrowdAttendance):
P(Successful | High)=0.8, P(Moderate | High)=0.15, P(Unsuccessful | High)=0.05
P(Successful | Medium)=0.4, P(Moderate | Medium)=0.5, P(Unsuccessful | Medium)=0.1
P(Successful | Low)=0.1, P(Moderate | Low)=0.3, P(Unsuccessful | Low)=0.6
P(MediaCoverage | EventSuccess):
P(Positive | Successful)=0.7, P(Neutral | Successful)=0.2, P(Negative | Successful)=0.1
P(Positive | Moderate)=0.3, P(Neutral | Moderate)=0.5, P(Negative | Moderate)=0.2
P(Positive | Unsuccessful)=0.1, P(Neutral | Unsuccessful)=0.3, P(Negative | Unsuccessful)=0.6
Task:

Assume a reasonable CPD for P(CrowdAttendance | WeatherCondition, Temperature, Wind).
Implement the Bayesian Network using pgmpy with DiscreteBayesianNetwork.
Compute the three inference queries."""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the network structure
model = DiscreteBayesianNetwork([
    ('WeatherCondition', 'CrowdAttendance'),
    ('Temperature', 'CrowdAttendance'),
    ('Wind', 'CrowdAttendance'),
    ('CrowdAttendance', 'EventSuccess'),
    ('EventSuccess','MediaCoverage')
])

# Step 2: Define CPDs
# P(Disease): 0=Flu, 1=Cold
cpd_weathercondition = TabularCPD(  #sunny cloudy rain
    variable='WeatherCondition', variable_card=3,
    values=[[0.5], 
            [0.3],
            [0.2]]  
)

cpd_temperature = TabularCPD(  #hot mild cold
    variable='Temperature', variable_card=3,
    values=[[0.4], 
            [0.4],
            [0.2]]  
)

cpd_wind = TabularCPD(  # low high
    variable='Wind', variable_card=2,
    values=[
        [0.6],  
        [0.4]   
    ],
)

cpd_crowdAttendance = TabularCPD( #high medium low
    variable='CrowdAttendance', variable_card=3,
     values=[
        [0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.7, 0.6, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.7, 0.6, 0.5],  # High
        [0.3, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2, 0.3],  # Medium
        [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2]   # Low
    ],
    evidence=['WeatherCondition','Temperature','Wind'], evidence_card=[3,3,2]
)

cpd_EventSuccess = TabularCPD(
    variable='EventSuccess', variable_card=3,
    values=[
        #h   #m  #l
        [0.8,0.4,0.1],     #successful
        [0.15,0.5,0.3],   #moderate
        [0.05,0.1,0.6]     #unsuccessful
    ],
    evidence=['CrowdAttendance'], evidence_card=[3]
)

cpd_MediaCoverage = TabularCPD(
    variable='MediaCoverage', variable_card=3,
    values=[
        #s   #m   #us
        [0.7,0.3,0.1],    #positive
        [0.2,0.5,0.3],   #neutral
        [0.1,0.2,0.6]     #negative
    ],
    evidence=['EventSuccess'], evidence_card=[3]
)


# Step 3: Add CPDs
model.add_cpds(cpd_weathercondition,cpd_temperature,cpd_wind,cpd_crowdAttendance,cpd_EventSuccess,cpd_MediaCoverage)

# Step 4: Verify
assert model.check_model(), "Model is invalid!"

# Step 5: Perform inference
inference = VariableElimination(model)

# Define your queries here
# Query 1: What is the probability of Low CrowdAttendance given WeatherCondition=Rainy and Wind=High?

result1 = inference.query(variables=['CrowdAttendance'], evidence={'WeatherCondition': 2, 'Wind': 1})
print("P(CrowdAttendance | WeatherCondition=Rainy, Wind=High):")
print(result1)

# Query 2:What is the probability of WeatherCondition=Sunny given EventSuccess=Successful?
result2 = inference.query(variables=['WeatherCondition'], evidence={'EventSuccess': 0})
print("P(WeatherCondition | EventSuccess=Successful):")
print(result2)


#What is the probability of MediaCoverage=Positive given CrowdAttendance=High and Temperature=Mild?
result3 = inference.query(variables=['MediaCoverage'], evidence={'CrowdAttendance': 0,'Temperature':1})
print("P(MediaCoverage | CrowdAttendance=High and Temperature=Mild):")
print(result3)