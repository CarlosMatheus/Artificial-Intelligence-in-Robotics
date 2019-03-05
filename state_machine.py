import random
import math
import pygame
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


def get_delta_time(initial_state_time):
    a = initial_state_time
    b = pygame.time.get_ticks()
    return (pygame.time.get_ticks() - initial_state_time) / 1000


def on_state_change(agent):
    agent.angular_speed = 0
    agent.linear_speed = 0


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        self.initial_state_time = pygame.time.get_ticks()
        self.state_duration = MOVE_FORWARD_TIME

    def check_transition(self, agent, state_machine):
        delta_time = get_delta_time(self.initial_state_time)
        if delta_time > self.state_duration:
            agent.behavior.change_state(MoveInSpiralState())
            on_state_change(agent)

    def execute(self, agent):
        agent.linear_speed = FORWARD_SPEED


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        self.initial_state_time = pygame.time.get_ticks()
        self.state_duration = MOVE_IN_SPIRAL_TIME

    def check_transition(self, agent, state_machine):
        delta_time = get_delta_time(self.initial_state_time)
        if delta_time > self.state_duration:
            agent.behavior.change_state(MoveForwardState())
            on_state_change(agent)

    def execute(self, agent):
        delta_time = get_delta_time(self.initial_state_time)
        radios = INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * delta_time
        agent.linear_speed = FORWARD_SPEED
        agent.angular_speed = agent.linear_speed / radios


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        self.initial_state_time = pygame.time.get_ticks()
        self.state_duration = GO_BACK_TIME

    def check_transition(self, agent, state_machine):
        delta_time = get_delta_time(self.initial_state_time)
        if delta_time > self.state_duration:
            agent.behavior.change_state(RotateState())
            on_state_change(agent)

    def execute(self, agent):
        agent.linear_speed = BACKWARD_SPEED


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        self.initial_state_time = pygame.time.get_ticks()
        self.state_duration = random.random() * TURN_AROUND_MAX_TIME

    def check_transition(self, agent, state_machine):
        delta_time = get_delta_time(self.initial_state_time)
        if delta_time > self.state_duration:
            agent.behavior.change_state(MoveForwardState())
            on_state_change(agent)
    
    def execute(self, agent):
        agent.angular_speed = ANGULAR_SPEED
