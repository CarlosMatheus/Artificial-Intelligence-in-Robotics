import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_b, upper_b, hyperparams, i):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        upper_bound_hyper = 1.5

        self.hyperparams = hyperparams
        self.lower_bound = lower_b
        self.upper_bound = upper_b * upper_bound_hyper
        self.idx = i

        num_of_params = len(self.lower_bound)

        position = [0]*num_of_params
        velocity = [0]*num_of_params

        max_bound = max(max(self.lower_bound), max(self.upper_bound))

        for idx in range(len(position)):
            position[idx] = random.uniform(self.lower_bound[idx], self.upper_bound[idx])
            velocity[idx] = random.uniform(-1*max_bound, max_bound)

        self.position = np.array(position)
        self.velocity = np.array(velocity)

        self.particle_best_position = self.position.copy()
        self.particle_best_value = -inf

    def update_particle_velocity(self, best_position):
        rp = random.uniform(0.0, 1.0)
        rg = random.uniform(0.0, 1.0)

        inertial_factor = self.hyperparams.inertia_weight * self.velocity
        cognitive_factor = self.hyperparams.cognitive_parameter * rp * (self.particle_best_position - self.position)
        social_factor = self.hyperparams.social_parameter * rg * (best_position - self.position)

        self.velocity = inertial_factor + cognitive_factor + social_factor

    def update_particle_position(self):
        new_position = self.position + self.velocity
        for idx in range(len(new_position)):
            if not self.lower_bound[idx] < new_position[idx] < self.upper_bound[idx]:
                self.velocity[idx] *= -1
                new_position[idx] = self.position[idx]
        self.position = new_position


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
        self._hyperparams = hyperparams
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._particles = [Particle(lower_bound, upper_bound, hyperparams, i) for i in range(hyperparams.num_particles)]
        self._best_value = -inf
        self._best_particle = None
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
        return particle.position

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
        actual_particle = self._particles[self._evaluate_idx]
        if value > actual_particle.particle_best_value:
            actual_particle.particle_best_value = value
            actual_particle.particle_best_position = actual_particle.position

        if value > self._best_value:
            self._best_value = value
            self._best_particle = actual_particle

        self._evaluate_idx += 1
        if self._evaluate_idx >= self._hyperparams.num_particles:
            self.advance_generation()
            self._evaluate_idx = 0

