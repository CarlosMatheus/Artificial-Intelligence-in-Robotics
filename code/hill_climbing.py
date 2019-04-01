from math import inf


def hill_climbing(cost_function, neighbors, theta0,
                  epsilon, max_iterations,
                  check_stopping_condition):
    """
    Executes the Hill Climbing (HC) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param neighbors: function which returns the neighbors of a given point.
    :type neighbors: list of numpy.array.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    theta = theta0
    history = [theta0]
    i = 0
    while not check_stopping_condition(max_iterations, i, cost_function, epsilon, theta):
        best = None  # J(None) = -inf
        for neighbor in neighbors(theta):
            # print(cost_function(neighbor))
            # print(cost_function(best))
            if cost_function(neighbor) < cost_function(best):
                best = neighbor
        if cost_function(best) > cost_function(theta):
            history.append(theta)
            # print(history)
            break
        theta = best
        history.append(theta)
        # print(history)
        i += 1

    return theta, history
