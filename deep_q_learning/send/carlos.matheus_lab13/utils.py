START_POSITION_CAR = -0.5


def reward_engineering_mountain_car(state, action, reward, next_state, done):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 2).
    :param action: action.
    :type action: int.
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 2).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :return: modified reward for faster training.
    :rtype: float.
    """

    r_original = reward

    position = state[0]
    velocity = state[1]

    next_position = next_state[0]
    start = START_POSITION_CAR

    r_modified = r_original + (position - start)**2 + velocity**2

    if next_position > 0.5:
        r_modified += 50

    return r_modified


