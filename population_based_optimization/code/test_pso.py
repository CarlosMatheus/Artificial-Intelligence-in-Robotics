import numpy as np
from particle_swarm_optimization import ParticleSwarmOptimization
from utils import Params
import matplotlib.pyplot as plt
from math import inf


def quality_function(x):
    # Defining a quadratic cost function. The optimum is this case is trivial: [1.0, 2.0, 3.0].
    return -((x[0] - 1.0) ** 2.0 + (x[1] - 2.0) ** 2.0 + (x[2] - 3.0) ** 2.0)


# Defining hyperparameters for the algorithm
hyperparams = Params()
hyperparams.num_particles = 40
hyperparams.inertia_weight = 0.7
hyperparams.cognitive_parameter = 0.6
hyperparams.social_parameter = 0.8
# Defining the lower and upper bounds
lower_bound = np.array([0.0, 0.0, 0.0])
upper_bound = np.array([3.0, 3.0, 3.0])
pso = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
position_history = []
quality_history = []
# Number of function evaluations will be 1000 times the number of particles,
# i.e. PSO will be executed by 1000 generations
num_evaluations = 1000 * hyperparams.num_particles
for i in range(num_evaluations):
    position = pso.get_position_to_evaluate()
    value = quality_function(position)
    pso.notify_evaluation(value)
    position_history.append(position)
    quality_history.append(value)
# Finally, print the best position found by the algorithm and its value
print('Best position:', pso.get_best_position())
print('Best value:', pso.get_best_value())

fig_format = 'png'
plt.figure()
plt.plot(position_history)
plt.legend(['x[0]', 'x[1]', 'x[2]'])
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameters Convergence')
plt.grid()
plt.savefig('test_parameters_converge.%s' % fig_format, format=fig_format)
plt.figure()
plt.plot(quality_history)
plt.xlabel('Iteration')
plt.ylabel('Quality')
plt.title('Quality Convergence')
plt.grid()
plt.savefig('test_quality_converge.%s' % fig_format, format=fig_format)
best_history = []
best = -inf
for q in quality_history:
    if q > best:
        best = q
    best_history.append(best)
plt.figure()
plt.plot(best_history)
plt.xlabel('Iteration')
plt.ylabel('Best Quality')
plt.title('Best Quality Convergence')
plt.grid()
plt.savefig('test_best_convergence.%s' % fig_format, format=fig_format)
plt.show()

f_1 = open("test_short_quality_history.txt", "w+")
f_2 = open("test_complete_quality_history.txt", "w+")
f_3 = open("test_short_position_history.txt", "w+")
f_4 = open("test_complete_position_history.txt", "w+")
window = 10
first = True
for idx in range(len(quality_history)):
    f_2.write("{:05d}".format(idx + 1) + ": " + str(quality_history[idx]) + "\n")
    if idx < window or idx >= len(quality_history) - window:
        f_1.write("{:05d}".format(idx + 1) + ": " + str('%.3f' % quality_history[idx]) + "\n")
    else:
        if first:
            f_1.write("..." + "\n")
            first = False
first = True
for idx in range(len(position_history)):
    f_4.write("{:05d}".format(idx + 1) + ": " + str(position_history[idx].tolist()) + "\n")
    if idx < window or idx >= len(position_history) - window:
        f_3.write("{:05d}".format(idx + 1) + ": " + str(['%.3f' % m for m in position_history[idx]]) + "\n")
    else:
        if first:
            f_3.write("..." + "\n")
            first = False
f_1.close()
f_2.close()
f_3.close()
f_4.close()
