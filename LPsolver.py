import math
import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers

# Theta is a I*F matrix
# C is the bound on number of files
# Return the optimal solution to the LP and is of the from J*F

# variables are order in this way:- z^1_1, z^2_1, ... z^J_1, i.e files are constant and we vary users similarly for y^j_f.

def SolveLP(Adj, Theta, C, start, t):
    
    I = np.shape(Adj)[0]
    J = np.shape(Adj)[1]
    F = np.shape(Theta)[1]
    
    
    A_Matrix = np.zeros(((I*F)+2*J,(I+J)*F)) #bigger Adj matrix for AX<=b'
    A_Matrix[0:I*F, 0:I*F] = np.eye(I*F)
    
    for i in range (0, F):
        A_Matrix[I*i: I*(i + 1) , I*F + J*i: I*F + J*(i + 1)] = -Adj
    
    # cache capacity equations
    for j in range(0, J):
        for f in range(0, F):
            A_Matrix[I*F + j][I*F + f*J + j] = 1 # upper bounding the cache capacity 
            A_Matrix[I*F + j + J][I*F + f*J + j] = -1 #enforcing the equality constraint

    b = np.zeros((1,I*F + 2*J))
    
    b[0:1, I*F:(I*F + J)] = C*np.ones((1,J))
    b[0:1, (I*F + J) :(I*F + 2*J)] = -C*np.ones((1,J))
    
    
    c =np.zeros((1, (I + J)*F)) # I*N array and then flatten it 
    c[0, 0:I*F] = -(Theta.T).reshape((1,I*F))
    
    A_Matrix = matrix(A_Matrix)
    b = matrix(b)
    c = matrix(c)
    
    
    #''' Using warm start

    #if t>0: # Using the warm start from the previous solution if t>=1
    #    res = linprog(c, A_ub=A_Matrix, b_ub=b, bounds=(0, 1), method = "revised simplex", x0=start)
    #else:
    #    res = linprog(c, A_ub=A_Matrix, b_ub=b,bounds=(0, 1))
    #'''

    # Not using the warm start
    res = linprog(c, A_ub=A_Matrix, b_ub=b,bounds=(0, 1), method="highs")

    #res = linprog(c, A_ub=A_Matrix, b_ub=b,bounds=(0, 1), method="highs", options={'lstsq':True,'cholesky':False, 'tol':1e-3})
    #solvers.lp(c, A_Matrix, b)
    
    Y = np.array(res['x'])[-J*F:]
    Y = Y.reshape((F, J)).T
    return (Y, res.x)
