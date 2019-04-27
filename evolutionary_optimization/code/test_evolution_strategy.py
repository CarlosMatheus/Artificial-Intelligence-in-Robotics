from simple_evolution_strategy import SimpleEvolutionStrategy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from benchmark import translated_sphere, ackley, schaffer2d, rastrigin
import cma


def animate(i):
    """
    Updates the optimization progress animation.
    :param i: current iteration.
    """
    plot_x.set_data(history_samples[i][:, 0], history_samples[i][:, 1])


def print_in_file(fitnesses, samples, i, num_iterations, file_history_complete, file_history_short):
    idxs = np.argsort(fitnesses)
    fit = fitnesses[idxs[0]]
    sample = samples[idxs[0]]

    begin = "{:03d}".format(i + 1) + ': '
    end = '\n'

    file_history_complete.write(begin + str(sample) + end)
    file_fitness_complete.write(begin + str(fit) + end)

    if not 10 <= i < num_iterations - 10:
        file_history_short.write(begin + str(sample) + end)
        file_fitness_short.write(begin + str(fit) + end)
    else:
        if i == 10:
            file_history_short.write('...' + end)
            file_fitness_short.write('...' + end)

# Evolution Strategy algorithm (Simple ES or CMA-ES)
# algorithm = 'ses'  # 'ses' or 'cmaes'
algorithms = ['ses', 'cmaes']
# Function which the optimization algorithm will optimize
# function = translated_sphere  # translated_sphere, ackley, schaffer2d, rastrigin
functions = [translated_sphere, translated_sphere, ackley, schaffer2d, rastrigin]
# function_name = 'translated_sphere' # translated_sphere, ackley, schaffer2d, rastrigin
fig_format = 'png'  # 'svg' (Word), 'eps' (Latex), 'png' (best compatibility/worst quality)

base_name = 'test_evolution_strategy'

for algorithm in algorithms:
    for function in functions:

        function_name = function.__name__

        name = base_name + '-' + function_name + '-' + algorithm.upper()

        # Simple ES parameters
        m0 = np.random.uniform(np.array([-3.0, -3.0]), np.array([3.0, 3.0]))  # initial guess used in the optimization algorithm
        C0 = np.identity(2)  # initial covariance (SES)
        mu = 12  # number of parents used for computing the mean and covariance of the next generation (SES)
        population_size = 24  # population size (SES)
        # CMA-ES parameters
        sigma0 = 1.0  # initial step size (CMA-ES)

        if algorithm == 'ses':
            es = SimpleEvolutionStrategy(m0, C0, mu, population_size)
        else:
            es = cma.CMAEvolutionStrategy(m0, 1.0)

        file_history_complete = open(name + '_history_sample_complete.txt', 'w+')
        file_history_short = open(name + '_history_sample_short.txt', 'w+')

        file_fitness_complete = open(name + '_fitness_complete.txt', 'w+')
        file_fitness_short = open(name + '_fitness_short.txt', 'w+')

        num_iterations = 200
        history_samples = []  # collect the samples of all iterations
        for i in range(num_iterations):
            samples = es.ask()
            # To avoid making the implementation of SES harder, I avoided following
            # the same interface of CMA-ES, thus I need to put an if here.
            if algorithm == 'ses':
                fitnesses = np.zeros(np.size(samples, 0))
                for j in range(np.size(samples, 0)):
                    fitnesses[j] = function(samples[j, :])
                es.tell(fitnesses)
                samples = es.ask()
                history_samples.append(samples)
            else:
                fitnesses = [function(sample) for sample in samples]
                es.tell(samples, fitnesses)
                # reshaping samples to be a numpy matrix
                reshaped_samples = np.zeros((len(samples), np.size(samples[0])))
                for j in range(len(samples)):
                    reshaped_samples[j, :] = samples[j]
                history_samples.append(reshaped_samples)

            print_in_file(fitnesses, samples, i, num_iterations, file_history_complete, file_history_short)

        gif_name = name + '.html'
        image_name = name + '.%s' % fig_format

        # Plotting a color map of the function
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set(xlim=(-3, 3), ylim=(-3, 3))
        x = np.arange(-3.0, 3.0, 0.01)
        y = np.arange(-3.0, 3.0, 0.01)
        [X, Y] = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(np.size(X, 0)):
            for j in range(np.size(X, 1)):
                Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
        ax.contourf(X, Y, Z)
        # Plotting the samples
        num_generations = len(history_samples)
        plot_x, = ax.plot(history_samples[0][:, 0], history_samples[0][:, 1], '.r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # Making the animation
        animation = FuncAnimation(fig, animate, interval=100, frames=num_generations)
        animation.save(gif_name, writer='imagemagick', fps=5)
        # plt.show()

        # Saving the last generation of the optimization algorithm
        plt.figure()
        plt.contourf(X, Y, Z)
        plt.plot(history_samples[-1][:, 0], history_samples[-1][:, 1], '.r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(image_name, fig_format=fig_format)

        file_history_complete.close()
        file_history_short.close()
