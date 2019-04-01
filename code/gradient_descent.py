

def gradient_descent(cost_function, gradient_function, theta0,
                     alpha, epsilon, max_iterations,
                     check_stopping_condition, create_history_file):
    """
    Executes the Gradient Descent (GD)
    algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param gradient_function: gradient of the cost function.
    :type gradient_function: function.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param alpha: learning rate.
    :type alpha: float.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """

    algorithm = "gradient_descent"

    theta = theta0
    history = [theta0]
    i = 0
    while not check_stopping_condition(max_iterations, i, cost_function, epsilon, theta):
        theta = theta - alpha * gradient_function(theta)
        history.append(theta)
        i += 1

    create_history_file(history, algorithm)

    return theta, history
