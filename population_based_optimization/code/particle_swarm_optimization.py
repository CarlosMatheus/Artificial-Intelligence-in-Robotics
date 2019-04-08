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
        # todo: reflexion
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
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._particles = [Particle(-100, 100, hyperparams) for _ in range(hyperparams.num_particles)]
        self._best_value = -inf
        self._best_particle = self._particles[0]
        self._evaluate_idx = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.
        :return: the best position.
        :rtype: numpy array.
        """
        return self._best_particle.position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.
        :return: value of the best position.
        :rtype: float.
        """
        return self._best_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        particle = self._particles[self._evaluate_idx]
        self._evaluate_idx += 1
        return particle

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        best_position = self.get_best_position()
        for particle in self._particles:
            particle.update_particle_velocity(best_position)
            particle.update_particle_position()

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        if value > self._best_value:
            self._best_value = value
            self._best_particle = self._particles[self._evaluate_idx]


