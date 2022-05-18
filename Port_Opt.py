# -*- coding: utf-8 -*-
"""
Created on Wed May 11 03:03:42 2022

@author: kiero
"""
from cvxopt import matrix, solvers
import numpy as np
P = matrix([[0.00517632, 0.00282111437461, 0.00245537875947,-0.00235058300372],
             [0.0, 0.00654799, 0.00519700602621, 0.00600485560224],
              [0.0, 0.0, 0.00361225, 0.00504216811596],
               [0.0, 0.0, 0.0, 0.02961345]])
q = matrix([0.0,0.0,0.0,0.0])
G = matrix([[-1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0, 0.0],
               [0.0, 0.0, 0.0, -1.0]])
h = matrix([0.0,0.0,0.0,0.0])
A = matrix([[1.0],
             [1.0],
              [1.0],
               [1.0]])
b = matrix([1.0])
P = (P + P.T) / 2
np_matrix = np.array(P)
print(np.linalg.eigvalsh(np_matrix))

sol = solvers.qp(P,q,G,h,A,b)
sol['x']
print(sol['status'])

print('the value of the primal objective function is', sol['primal objective'])
print('the value of the dual objective function is', sol['dual objective'])
print('the value of the slackness condition is', sol['gap'])

print(sol['x'])
