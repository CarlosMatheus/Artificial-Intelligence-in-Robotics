# Path Finder

## Run the code

Make sure to have python3 installed and pip.

Setup the environment:
```bash
cd Artificial-Intelligence-in-Robotics
python3 -m venv venv
source venv/bin/activate
pip install requirements.txt
```

Run:
```bash
cd Artificial-Intelligence-in-Robotics/state_management/code
python simulation.py
```

## Some results

### State Machine

A finite automaton (AF) or finite state machine is a computational model that performs computations automatically on an input to produce an output.

This model is made up of an alphabet, a set of finite states, a transition function, an initial state and a set of final states. Its operation is based on a transition function, which receives from a starting state a string of characters belonging to the alphabet (the input), and which is reading that string as the automaton moves from one state to another, to finally stop at a final state or acceptance, which represents the exit.

The purpose of the finite automatons is to recognize regular languages, which correspond to the simplest formal languages according to the Chomsky Hierarchy.

![Alt text](report/imgs/stateMachine/5.png?raw=true "Title")

### Behavior Tree

Behavior Tree (BT) is a mathematical model of plan execution used in computer science, robotics, control systems, and video games. They describe the changes between a finite set of tasks in a modular way. Its strength comes from the ability to create very complex tasks, composed of simple tasks, without worrying about how simple tasks are implemented. The BTs have some similarities with the machines of hierarchical states, with the main difference that the main building block of a behavior is a task and not a state. Its ease of human understanding makes BTs less error prone and very popular in the gaming community. BTs have been shown to generalize several other control architectures.

![Alt text](report/imgs/decisionTree/7.png?raw=true "Title")
