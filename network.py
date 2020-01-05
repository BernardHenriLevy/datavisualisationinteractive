
from pomegranate import *
import json



### PARAMETRAGE DES NOEUDS
cloudy_parameters = {'V': 0.5, 'F': 0.5}
cloudy = DiscreteDistribution(cloudy_parameters)

sprinkler_parameters = [['V', 'V',  0.1],['V', 'F',  0.9],['F', 'F', 0.5],['F', 'V',  0.5]]
sprinkler = ConditionalProbabilityTable(sprinkler_parameters, [cloudy])


rain_parameters = [['V', 'V',  0.8],['V', 'F',  0.2],['F', 'F', 0.8],['F', 'V',  0.2]]
rain = ConditionalProbabilityTable(rain_parameters, [cloudy])


wetgrass_parameters =  [['V', 'V','V',  0.99],['V', 'V','F',  0.01],['V', 'F','V',  0.9],['V', 'F','F',  0.1],['F', 'V','V',  0.9],['F', 'V','F',  0.1],['F', 'F','V',  0],['F', 'F','F' , 1],]
wetgrass = ConditionalProbabilityTable(wetgrass_parameters, [sprinkler,rain])

d1 = State(cloudy, name="Cloudy")
d2 = State(sprinkler, name="Sprinkler")
d3 = State(rain,  name="Rain")
d4 = State(wetgrass,  name="Wetgrass")


network = BayesianNetwork("Prédiction")
network.add_states(d1, d2, d3,d4)
network.add_edge(d1, d2)
network.add_edge(d1, d3)
network.add_edge(d2, d4)
network.add_edge(d3, d4)

network.bake()


with open('wetgrass_network.json', 'w') as outfile:
    json.dump(network.to_json(), outfile)
