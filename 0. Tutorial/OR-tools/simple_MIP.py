# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:58:34 2021

@author: USER

Ref: https://github.com/google/or-tools/blob/stable/ortools/linear_solver/samples/simple_mip_program.py
(https://developers.google.com/optimization/examples)
"""

from ortools.linear_solver import pywraplp

# 1. Define solover
solver = pywraplp.Solver.CreateSolver('SCIP')


# 2. Defining variables
infinity = solver.infinity()

x = solver.IntVar(0.0, infinity, 'x') #Integer (0, infinit)
y = solver.IntVar(0.0, infinity, 'y')

print('Number of variables =', solver.NumVariables())


# 3. Define Constraints
solver.Add(x + 7 * y <= 17.5)
solver.Add(x <= 3.5)

print('Number of constraints =', solver.NumConstraints())

# 4. Define Objective Function
solver.Maximize(x + 10 * y)

# 5. Solve
status = solver.Solve()

# 6. Check the solution
if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())
else:
    print('The problem does not have an optimal solution.')
    
# 7. Check the model performance
print('\nAdvanced usage:')
print('Problem solved in %f milliseconds' % solver.wall_time())
print('Problem solved in %d iterations' % solver.iterations())
print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
