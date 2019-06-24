import numpy as np


def compute_greedy_policy_as_table(q):
    """
    Computes the greedy policy as a table.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :return: greedy policy table.
    :rtype: bidimensional numpy array.
    """
    policy = np.zeros(q.shape)
    for s in range(q.shape[0]):
        policy[s, greedy_action(q, s)] = 1.0
    return policy


def epsilon_greedy_action(q, state, epsilon):
    """
    Computes the epsilon-greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: int.
    :param epsilon: probability of selecting a random action.
    :type epsilon: float.
    :return: epsilon-greedy action.
    :rtype: int.
    """
    if state is None:
        print(1)
        print(state)


    if np.random.binomial(1, epsilon) == 1:
        # try:
        action = np.random.choice([a for a in range(len(q[state]))])
        # except:
        #     print(q[state])
        #     print(state)
        if action < 0:
            print('aqui1')
    else:
        values_ = q[state]
        # try:
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # except:
        #     print(np.max(values_))
        # print(action)
        if action < 0:
            print('aqui')

    a = action
    action = int(action)
    if action == -1:
        print(a)
        print('asdf')

    return int(action)


def greedy_action(q, state):
    """
    Computes the greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: int.
    :return: greedy action.
    :rtype: int.
    """
    values_ = q[state]
    return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


class RLAlgorithm:
    """
    Represents a model-free reinforcement learning algorithm.
    """
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        """
        Creates a model-free reinforcement learning algorithm.

        :param num_states: number of states of the MDP.
        :type num_states: int.
        :param num_actions: number of actions of the MDP.
        :type num_actions: int.
        :param epsilon: probability of selecting a random action in epsilon-greedy policy.
        :type epsilon: float.
        :param alpha: learning rate.
        :type alpha: float.
        :param gamma: discount factor.
        :type gamma: float.
        """
        self.q = np.zeros((num_states, num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_num_states(self):
        """
        Returns the number of states of the MDP.

        :return: number of states.
        :rtype: int.
        """
        return self.q.shape[0]

    def get_num_actions(self):
        """
        Returns the number of actions of the MDP.

        :return: number of actions.
        :rtype: int.
        """
        return self.q.shape[1]

    def get_exploratory_action(self, state):
        """
        Returns an exploratory action using epsilon-greedy policy.

        :param state: current state.
        :type state: int.
        :return: exploratory action.
        :rtype: int.
        """
        if state is None:
            print(state)
        return epsilon_greedy_action(self.q, state, self.epsilon)

    def get_greedy_action(self, state):
        """
        Returns a greedy action considering the policy of the RL algorithm.

        :param state: current state.
        :type state: int.
        :return: greedy action considering the policy of the RL algorithm.
        :rtype: int.
        """
        raise NotImplementedError('Please implement this method')

    def learn(self, state, action, reward, next_state, next_action):
        raise NotImplementedError('Please implement this method')


class Sarsa(RLAlgorithm):
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        super().__init__(num_states, num_actions, epsilon, alpha, gamma)

    def get_greedy_action(self, state):
        """
        Notice that Sarsa is an on-policy algorithm, so it uses the same epsilon-greedy
        policy for learning and execution.

        :param state: current state.
        :type state: int.
        :return: epsilon-greedy action of Sarsa's execution policy.
        :rtype: int.
        """
        if state is None:
            print(state)
        return epsilon_greedy_action(self.q, state, self.epsilon)

    def learn(self, state, action, reward, next_state, next_action):
        target = 0.0
        q_next = self.q[next_state]
        best_actions = np.argwhere(q_next == np.max(q_next))
        for action_ in self.q[state]:
            action_ = int(action_)
            if action_ in best_actions:
                # print(state)
                # print(next_state)
                # print(action_)
                # print(best_actions)
                target += ((1.0 - self.epsilon) / len(best_actions) + self.epsilon / len(self.q[state])) * self.q[next_state][action_]
            else:
                target += self.epsilon / len(self.q[state]) * self.q[next_state][action_]
        target *= self.gamma
        action = int(action)
        self.q[state][action] += self.alpha * (reward + target - self.q[state][action])


class QLearning(RLAlgorithm):
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        super().__init__(num_states, num_actions, epsilon, alpha, gamma)

    def get_greedy_action(self, state):
        return greedy_action(self.q, state)

    def learn(self, state, action, reward, next_state, next_action):
        self.q[state][action] += self.alpha * (reward + self.gamma * np.max(self.q[next_state]) - self.q[state][action])
