# Mathematical Optimization

Three optimization methods have been studied and implemented that have algorithms based on local search. These methods were Gradient Descent, Hill Climbing and Simulated Anneling.

The methods were tested in a problem proposed in the script of this activity. This problem utilized linear regression for ober physical parameters related to the movement of a ball.

Since the treated problem has an analytical solution, the experiment is an educational one, since its solution can be easily obtained by the least squares method (MMQ).

In general, the activity has been successfully developed. In each of the following sessions are the respective development of one of the methods studied.

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
cd Artificial-Intelligence-in-Robotics/mathematical_optimization/code
python ball_fit.py
```

## Some results

### Gradient Descent

The gradient method is a numerical method used in optimization. To find a minimum (local) of a function we use an iterative scheme, where in each step we take the (negative) direction of the gradient, which corresponds to the direction of maximum slope. It can be regarded as the method followed by a course of water, in its descent by the force of gravity.

![Alt text](code/gradient_descent.png?raw=true "Title")

### Hill Climbing

Also called the Simple Climbing Algorithm or Hill Climb Algorithm is a mathematical optimization technique that belongs to the family of local search algorithms. It is an iterative algorithm that begins with an arbitrary solution to a problem, then tries to find a better solution by incrementally varying a single element of the solution. If the change produces a better solution, another incremental change is made to the new solution, repeating this process until improvements can not be found. This voracious algorithm is usually called local, because it takes a "good" neighbor state without thinking about the next action.

![Alt text](code/hill_climbing.png?raw=true "Title")

### Simulated Annealing

Simulated annealing is a meta-heuristic search algorithm for global optimization problems; The general objective of this type of algorithm is to find a good approximation to the optimal value of a function in a large search space. This optimal value is called "global optimum".

![Alt text](code/simulated_annealing.png?raw=true "Title")

### Comparison

![Alt text](code/optimization_comparison.png?raw=true "Title")
![Alt text](code/fit_comparison.png?raw=true "Title")
