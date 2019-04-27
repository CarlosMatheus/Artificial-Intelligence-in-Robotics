import numpy as np


class SimpleEvolutionStrategy:
    """
    Represents a simple evolution strategy optimization algorithm.
    The mean and covariance of a gaussian distribution are evolved at each generation.
    """
    def __init__(self, m0, C0, mu, population_size):
        """
        Constructs the simple evolution strategy algorithm.

        :param m0: initial mean of the gaussian distribution.
        :type m0: numpy array of floats.
        :param C0: initial covariance of the gaussian distribution.
        :type C0: numpy matrix of floats.
        :param mu: number of parents used to evolve the distribution.
        :type mu: int.
        :param population_size: number of samples at each generation.
        :type population_size: int.
        """
        self.m = m0
        self.C = C0
        self.mu = mu
        self.population_size = population_size
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)
        # print(self.samples)

    def ask(self):
        """
        Obtains the samples of this generation to be evaluated.
        The returned matrix has dimension (population_size, n), where n is the problem dimension.

        :return: samples to be evaluated.
        :rtype: numpy array of floats.
        """
        return self.samples

    def tell(self, fitnesses):
        """
        Tells the algorithm the evaluated fitnesses. The order of the fitnesses in this array
        must respect the order of the samples.

        :param fitnesses: array containing the value of fitness of each sample.
        :type fitnesses: numpy array of floats.
        """
        # print('==================================')
        # print(fitnesses, self.samples)
        # best_fit = 10000000000
        # best_sample = None
        # for idx in range(len(fitnesses)):
        #     fitness = fitnesses[idx]
        #     if fitness < best_fit:
        #         best_fit = fitness
        #         best_sample = self.samples[idx]

        # combined = zip(fitnesses, self.samples)

        # combined.sort()

        # fitnesses = [c[0] for c in combined]
        # self.samples = [c[1] for c in combined]
        indices = np.argsort(fitnesses)
        # print(fitnesses)
        # print(indices)
        best_samples = self.samples[indices[0:self.mu], :]
        # print(best_samples)
        # print(best_samples)

        # self.C = sum(self.samples[:self.mu])/self.mu
        # array = np.zeros(np.size(self.C))
        array = self.C * 0
        # print(array)
        for idx in range(self.mu):
            mt = best_samples[idx] - self.m
            mt = np.matrix(mt)
            transpose = mt.transpose()
            # print(mt * transpose)
            array += (transpose*mt)
        # print(array)
        self.C = array / self.mu
        # print(self.C)

        # self.m = sum(self.samples[:self.mu])/self.mu
        self.m = sum(best_samples)/self.mu

        # print(best_sample)
        # print(self.m)
        # print(self.C)
        self.samples = np.random.multivariate_normal(self.m, self.C, self.population_size)
