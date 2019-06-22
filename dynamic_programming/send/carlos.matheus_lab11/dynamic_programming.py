import numpy as np
from math import inf, fabs
from utils import *


def random_policy(grid_world):
    """
    Creates a random policy for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :return: random policy.
    :rtype: tridimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    policy = (1.0 / NUM_ACTIONS) * np.ones((dimensions[0], dimensions[1], NUM_ACTIONS))
    return policy


def greedy_policy(grid_world, value, epsilon=1.0e-3):
    """
    Computes a greedy policy considering a value function for a grid world. If there are more than
    one optimal action for a given state, then the optimal action is chosen at random.


    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param value: the value function.
    :type value: bidimensional NumPy array.
    :param epsilon: tolerance used to consider that more than one action is optimal.
    :type epsilon: float.
    :return: greedy policy.
    :rtype: tridimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    policy = np.zeros((dimensions[0], dimensions[1], NUM_ACTIONS))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            current_state = (i, j)
            if not grid_world.is_cell_valid(current_state):
                # Assuming random action if the cell is an obstacle
                policy[i, j] = (1.0 / NUM_ACTIONS) * np.ones(NUM_ACTIONS)
                continue
            max_value = -inf
            action_value = np.zeros(NUM_ACTIONS)  # Creating a temporary q(s, a)
            for action in range(NUM_ACTIONS):
                r = grid_world.reward(current_state, action)
                action_value[action] = r
                for next_state in grid_world.get_valid_sucessors((i, j), action):
                    transition_prob = grid_world.transition_probability(current_state, action, next_state)
                    action_value[action] += grid_world.gamma * transition_prob * value[next_state[0], next_state[1]]
                if action_value[action] > max_value:
                    max_value = action_value[action]
            # This post-processing is necessary since we may have more than one optimal action
            num_actions = 0
            for action in range(NUM_ACTIONS):
                if fabs(max_value - action_value[action]) < epsilon:
                    policy[i, j, action] = 1.0
                    num_actions += 1
            for action in range(NUM_ACTIONS):
                policy[i, j, action] /= num_actions
    return policy


def evaluate(grid_world, initial_value, policy, dimensions, num_iterations, epsilon):

    old_value = initial_value
    value = np.copy(initial_value)

    for _ in range(num_iterations):
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_state = (i, j)
                possible_actions = policy[current_state[0]][current_state[1]]
                sum_val = 0
                for action in range(len(possible_actions)):
                    rew = grid_world.reward(current_state, action)
                    successors_states = grid_world.get_valid_sucessors(current_state)
                    val_sum = 0
                    for successor in successors_states:
                        prob = grid_world.transition_probability(current_state, action, successor)
                        val = old_value[successor[0]][successor[1]]
                        val_sum += prob * val
                    sum_val += (rew + grid_world.gamma * val_sum) * possible_actions[action]

                value[current_state[0]][current_state[1]] = sum_val

        if changed_val(old_value, value, dimensions, epsilon):
            old_value = value
            value = np.copy(old_value)
        else:
            break

    return value


def changed_val(old_value, new_value, dimensions, epsilon):
    """
    Checks whether the value changed some of its elements values or not
    :param old_value: the old value of the last iteration
    :param new_value: the new value of the actual iteration
    :param dimensions: The dimensions of the grid
    :param epsilon: the threshold in which two numbers are considered equals
    :return: boolean telling if occurred a change of value
    """
    for j in range(dimensions[0]):
        for i in range(dimensions[1]):
            if abs(new_value[i][j] - old_value[i][j]) > epsilon:
                return True
    return False


def policy_evaluation(grid_world, initial_value, policy, num_iterations=10000, epsilon=1.0e-5):
    """
    Executes policy evaluation for a policy executed on a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bidimensional NumPy array.
    :param policy: policy to be evaluated.
    :type policy: tridimensional NumPy array.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return: value function of the given policy.
    :rtype: bidimensional NumPy array.
    """

    dimensions = grid_world.dimensions
    value = evaluate(grid_world, initial_value, policy, dimensions, num_iterations, epsilon)
    return value


def value_iteration(grid_world, initial_value, num_iterations=10000, epsilon=1.0e-5):
    """
    Executes value iteration for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bi-dimensional NumPy array.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return value: optimal value function.
    :rtype value: bi-dimensional NumPy array.
    """

    dimensions = grid_world.dimensions
    possible_actions = [STOP, UP, RIGHT, DOWN, LEFT]
    old_value = initial_value
    value = np.copy(initial_value)

    for _ in range(num_iterations):
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_state = (i, j)
                max_val = -float('inf')

                for action in possible_actions:
                    rew = grid_world.reward(current_state, action)
                    successors_states = grid_world.get_valid_sucessors(current_state)
                    val_sum = 0
                    for successor in successors_states:
                        prob = grid_world.transition_probability(current_state, action, successor)
                        val = old_value[successor[0]][successor[1]]
                        val_sum += prob * val
                    max_val = max(max_val, rew + grid_world.gamma * val_sum)

                value[current_state[0]][current_state[1]] = max_val

        if changed_val(old_value, value, dimensions, epsilon):
            old_value = value
            value = np.copy(old_value)
        else:
            break

    return value


def is_policy_equal(old_policy, new_policy):
    """
    Check whether a new policy is equal to the old policy
    :param old_policy: the old policy
    :param new_policy: the new policy
    :return: boolean true if equal
    """
    return np.array_equal(old_policy, new_policy)


def policy_iteration(grid_world, initial_value, initial_policy, evaluations_per_policy=3, num_iterations=10000,
                     epsilon=1.0e-5):
    """
    Executes policy iteration for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bidimensional NumPy array.
    :param initial_policy: initial policy used to bootstrap the algorithm.
    :type initial_policy: tridimensional NumPy array.
    :param evaluations_per_policy: number of policy evaluations per policy iteration.
    :type evaluations_per_policy: int.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return value: value function of the optimal policy.
    :rtype value: bidimensional NumPy array.
    :return policy: optimal policy.
    :rtype policy: tridimensional NumPy array.
    """

    value = np.copy(initial_value)
    policy = np.copy(initial_policy)

    for i in range(num_iterations):
        value = policy_evaluation(grid_world, value, policy, evaluations_per_policy, epsilon)
        new_policy = greedy_policy(grid_world, value)

        if not is_policy_equal(policy, new_policy):
            policy = new_policy
        else:
            break

    return value, policy

