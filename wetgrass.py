import math
from pomegranate import *

cloudy = DiscreteDistribution({'V': 0.5, 'F': 0.5})



sprinkler = ConditionalProbabilityTable(
    [['V', 'V',  0.1],
     ['V', 'F',  0.9],
     ['F', 'F', 0.5],
     ['F', 'V',  0.5]
     ], [cloudy])

rain = ConditionalProbabilityTable(
    [['V', 'V',  0.8],
     ['V', 'F',  0.2],
     ['F', 'F', 0.8],
     ['F', 'V',  0.2]
     ], [cloudy])



wetgrass = ConditionalProbabilityTable(
    [
     ['V', 'V','V',  0.99],['V', 'V','F',  0.01],
     ['V', 'F','V',  0.9],['V', 'F','F',  0.1],
     ['F', 'V','V',  0.9],['F', 'V','F',  0.1],
     ['F', 'F','V',  0],['F', 'F','F' , 1],
     ], [sprinkler,rain])

d1 = State(cloudy, name="cloudy")
d2 = State(sprinkler, name="sprinkler")
d3 = State(rain,  name="rain")
d4 = State(wetgrass,  name="wetgrass")


network = BayesianNetwork("Prédiction")
network.add_states(d1, d2, d3,d4)
network.add_edge(d1, d2)
network.add_edge(d1, d3)
network.add_edge(d2, d4)
network.add_edge(d3, d4)

network.bake()

beliefs = network.predict_proba({'sprinkler': 'V'})
beliefs = map(str, beliefs)
print("Probabilité ".join("{}{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))