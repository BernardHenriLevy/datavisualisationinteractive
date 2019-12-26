import math
from pomegranate import *

ey = DiscreteDistribution({'Bl': 0.36, 'Br': 0.37, 'Gr': 0.10, 'Ha': 0.15})



br = ConditionalProbabilityTable(
    [['Bl', 'Bla',  0.09],
     ['Bl', 'Blo',  0.43],
     ['Bl', 'Bro',  0.39],
     ['Bl', 'Re',  0.07],
     ['Br', 'Bla', 0.30],
     ['Br', 'Blo', 0.03],
     ['Br', 'Bro', 0.54],
     ['Br', 'Re', 0.11],
     ['Gr', 'Bla', 0.07],
     ['Gr', 'Blo', 0.25],
     ['Gr', 'Bro', 0.45],
     ['Gr', 'Re', 0.21],
     ['Ha', 'Bla', 0.16],
     ['Ha', 'Blo', 0.10],
     ['Ha', 'Bro', 0.57],
     ['Ha', 'Re', 0.15]], [ey])



rb = ConditionalProbabilityTable(
    [['Bla', 'O',  0.25],
     ['Blo', 'O',  0.33],
     ['Bro', 'O',  0.10],
     ['Re', 'O',  0.5],
     ['Bla', 'N', 0.75],
     ['Blo', 'N', 0.67],
     ['Bro', 'N', 0.90],
     ['Re', 'N', 0.5],
     ], [br])

d1 = State(ey, name="ey")
d2 = State(br, name="br")
d3 = State(rb,  name="rb")

network = BayesianNetwork("Prédiction Ey->Br")
network.add_states(d1, d2, d3)
network.add_edge(d1, d2)
network.add_edge(d2, d3)

network.bake()

beliefs = network.predict_proba({'rb': 'O'})
beliefs = map(str, beliefs)
print("Probabilité ".join("{}{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))