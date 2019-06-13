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


def evaluate(grid_world, value, policy, dimensions):
    new_value = np.copy(initial_value)

    return new_value


def diff_in_val_greater_than_epsilon(old_value, new_value, dimensions, epsilon):
    for j in range(dimensions[0]):
        for i in range(dimensions[1]):
            if abs(new_value[i][j] - old_value[i][j]) < epsilon:
                return False
    return True


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
    """
    Intera chamando value interaion? 
    até convergir o value e entao retornando aquela policy evaluated
    """
    # Todo: Test
    dimensions = grid_world.dimensions
    value = np.copy(initial_value)

    for i in range(num_iterations):
        new_value = evaluate(grid_world, value, dimensions)
        if not diff_in_val_greater_than_epsilon(value, new_value, dimensions, epsilon):
            value = new_value
        else:
            break

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
    """
    executes just one value iteration???
    Will use the bellman equation
    the value iteration, 
    initialize all grid with -1, and end with 0
    iterate using bellman, not makaing recursion, but using the previos genaratiiin board
    after a large number of iterations, the values inside the grid will converge
    the CORRECT_ACTION_PROB and GAMMA are inside grid_world'
    ja existe a funcao transition_probability => p(s'|s, a)
    ja existe a funcao reward => r(s, a)
    ja existe get_valid_sucessors
    """

    # Todo: Test this function

    dimensions = grid_world.dimensions
    value = np.copy(initial_value)

    possible_actions = [STOP, UP, RIGHT, DOWN, LEFT]

    old_value = value
    value = np.copy(old_value)

    for _ in range(num_iterations):
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_state = (i, j)
                max_val = -float('inf')
                for action in possible_actions:
                    rew = grid_world.reward(current_state, action)
                    sucessors_states = grid_world.get_valid_sucessors(current_state)
                    val_sum = 0
                    for sucessor in sucessors_states:
                        prob = grid_world.transition_probability(current_state, action, sucessor)
                        val = old_value[sucessor[0]][sucessor[1]]
                        val_sum += prob * val
                    max_val = max(max_val, rew + grid_world.gamma * val_sum)

                value[current_state[0]][current_state[1]] = max_val

        if diff_in_val_greater_than_epsilon(old_value, value, dimensions, epsilon):
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

    """
    alterna entre (avaliacao de politica) e (aprimoramento de politica) 
    o aprimoramento eh realizado atravws de tomar uma politica gulosa a partir da funcao valor da autal politirca ?? 
    pode-se mostrar que esse alg converge pra uma politica otima
    for:
        intera chamando police evaluation e depois police improvement (greedy) até que convergimos para a melhor police e melhor value funciotn
    intera ate convergir
    """
    # Todo: Check if it is correct

    value = np.copy(initial_value)
    policy = np.copy(initial_policy)

    for i in range(num_iterations):
        new_value = policy_evaluation(grid_world, value, policy)
        new_policy = greedy_policy(grid_world, new_value)
        if not is_policy_equal(policy, new_policy):
            policy = new_policy
            value = new_value
        else:
            break

    return value, policy

