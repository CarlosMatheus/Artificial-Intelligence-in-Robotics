from math import exp
import random


def simulated_annealing(cost_function, random_neighbor, schedule,
                        theta0, epsilon, max_iterations,
                        check_stopping_condition, create_history_file):
    """
    Executes the Simulated Annealing (SA) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param random_neighbor: function which returns a random neighbor of a given point.
    :type random_neighbor: numpy.array.
    :param schedule: function which computes the temperature schedule.
    :type schedule: function.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: np.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of np.array.
    """
    theta = theta0
    history = [theta0]
    i = 0

    algorithm = "simulated_annealing"

    while not check_stopping_condition(max_iterations, i, cost_function, epsilon, theta):
        T = schedule(i)
        # print(T)
        if T < 0.0:
            return theta, history
        neighbor = random_neighbor(theta)
        deltaE = cost_function(neighbor) - cost_function(theta)
        if deltaE < 0:
            theta = neighbor
        else:
            r = random.uniform(0.0, 1.0)
            if r >= exp(-deltaE / T):
                # print(i, "yes")
                theta = neighbor
            # else:
                # print(i, "no")

        history.append(theta)
        i += 1

    # print(history)
    create_history_file(history, algorithm)

    return theta, history
