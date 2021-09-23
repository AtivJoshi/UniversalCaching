import math
import numpy as np
from Phi import phi  # takes in arguments Adj and X(cache configuration)

# X is a feasible solution given in a matrix form (J, F) 
# Theta and Adj are useful to calculate phi
# Theta is of the form (I, F) and adj is of the form (I, J)


def madow_rounding(X, Theta, Adj, C):
    
    A = np.argwhere(X > 0).tolist()# gets the non-zero entries into a list, which is sorted by the first co-ordinate.
    to_round = len(A) #size of the list, to make the code faster
    
    #because there are errors in the LPsolver we need to be careful if there is only one file 
    #which is fractional over a single cache

    X = np.minimum(X, np.ones_like(X)) # because some X^i_f might have been filled more due to errors in precision
    J, F = np.shape(X)
    X_r = np.zeros((J, F))
    Pi = np.zeros((J, F + 1))
    Pi[:, 1: ] = np.cumsum(X, axis= 1) #the cumulativate probabilities 
    U = np.random.rand(J)
    
    
    cache = 0
    while(cache < J):

        for k in range(C):
            file = 0
            while(Pi[cache][file] <= U[cache] + k):
                file += 1
                if(file == F):
                    break
            #print(cache, file)
            X_r[cache][file - 1] = 1

        cache += 1

    #print("Madow")
    #print(Pi)
    #print(U)
    #print(X, X_r)
    #print("-______________________")
    return X_r