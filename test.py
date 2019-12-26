
import math
from pomegranate import *


#A = Bon
#B = Moyen
#C= Faible

competence =DiscreteDistribution( { 'A': 1./2, 'B': 1./4, 'C': 1./4 } )
 
machine =DiscreteDistribution( { 'A': 5./8, 'B': 2./8, 'C': 1./8 } )
 
panne = ConditionalProbabilityTable(
[[ 'A', 'A', 'N', 1.0 ],
[ 'A', 'A', 'O', 0.0 ],
[ 'A', 'B', 'N', 0.8 ],
[ 'A', 'B', 'O', 0.2 ],
[ 'A', 'B', 'N', 0.8 ],
[ 'A', 'C', 'O', 0.6 ],
[ 'A', 'C', 'N', 0.4 ],
[ 'B', 'A', 'N', 0.0 ],
[ 'B', 'A', 'O', 0.0 ],
[ 'B', 'A', 'N', 1.0 ],
[ 'B', 'B', 'O', 0.5 ],
[ 'B', 'B', 'O', 0.0 ],
[ 'B', 'B', 'N', 0.5 ],
[ 'B', 'C', 'N', 0.2 ],
[ 'B', 'C', 'O', 0.8 ],
[ 'C', 'A', 'N', 0.7 ],
[ 'C', 'A', 'O', 0.3 ],
[ 'C', 'B', 'O', 1.0 ],
[ 'C', 'B', 'N', 0.0 ],
[ 'C', 'C', 'O', 0.95 ],
[ 'C', 'C', 'N', 0.05 ]], [competence, machine] )
 
d1 = State( competence, name="competence" )
d2 = State( machine, name="machine" )
d3 = State( panne, name="panne" )


network = BayesianNetwork( "Prédiction de panne" )
network.add_states(d1, d2, d3)
network.add_edge(d1, d3)
network.add_edge(d2, d3)
network.bake()


beliefs = network.predict_proba({ 'competence': 'C'})
beliefs = map(str, beliefs)
print("Probabilité ".join( "{}{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))