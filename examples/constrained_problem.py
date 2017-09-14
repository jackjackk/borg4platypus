from platypus import NSGAII, Problem, Real
from borg4platypus import ExternalBorgC
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

def belegundu(vars):
    x = vars[0]
    y = vars[1]
    return [-2*x + y, 2*x + y], [-x + y - 1, x + y - 7]

problem = Problem(2, 2, 2)
problem.types[:] = [Real(0, 5), Real(0, 3)]
problem.constraints[:] = "<=0"
problem.function = belegundu

algorithm = ExternalBorgC(problem, 0.01)
#algorithm = NSGAII(problem)
algorithm.run(10000)

# display the results
plt.scatter(*[[s.objectives[i] for s in algorithm.result] for i in range(2)])
plt.show()