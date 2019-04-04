import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound, hyperparams):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        self.hyperparams = hyperparams

        num_of_params = 3

        position = []*num_of_params
        velocity = []*num_of_params

        for idx in range(len(position)):
            position[idx] = random.uniform(lower_bound, upper_bound)
            velocity[idx] = random.uniform(lower_bound/100, upper_bound/100)

        self.position = np.array(position)
        self.velocity = np.array(velocity)

        self.particle_best_position = self.position.copy()

    def update_particle_velocity(self, best_position):
        rp = random.uniform(0.0, 1.0)
        rg = random.uniform(0.0, 1.0)

        self.velocity = self.hyperparams.inertia_weight * self.velocity + \
                        self.hyperparams.cognitive_parameter*rp(
                            self.particle_best_position - self.position) + \
                        self.hyperparams.social_parameter*rg*(
                            best_position - self.position)

    def update_particle_position(self):
        self.position = self.position + self.velocity

class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        num_particles: num of particles
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.particles = [Particle(-100, 100)]*hyperparams.num_particles
        self.get_best_position()

        return

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.lower_bound  # Remove this line

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """

        # Todo: implement
        return 0.0  # Remove this line

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        best_postition = self.get_best_position()

        # 𝐯 = 𝜔𝐯 + 𝜑𝑟(𝐛−𝐱) + 𝜑𝑟(𝐛−𝐱)

        return self.lower_bound  # Remove this line

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        # Todo: implement
        pass  # Remove this line

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        pass  # Remove this line

