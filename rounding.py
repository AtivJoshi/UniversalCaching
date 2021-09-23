import math
import numpy as np
from Phi import phi  # takes in arguments Adj and X(cache configuration)

# X is a feasible solution given in a matrix form (J, F) 
# Theta and Adj are useful to calculate phi
# Theta is of the form (I, F) and adj is of the form (I, J)


def cache_by_cache_pipage_rounding(X, Theta, Adj, C):
    
    A = np.argwhere(X > 0).tolist()# gets the non-zero entries into a list, which is sorted by the first co-ordinate.
    to_round = len(A) #size of the list, to make the code faster
    
    #because there are errors in the LPsolver we need to be careful if there is only one file 
    #which is fractional over a single cache
    
    cache_fill = np.sum(X, axis = 1) #calculates how much of the capacity is actually filled because the factional solutions might not fill it up totally.
    while(to_round > 1): # that is there are more one cache,file pair to round
        
        if(A[-1][0] == A[-2][0]): # to check if both (cahe, file) pair belong to the same cache
            
            [j, file_1] = A.pop()
            [j, file_2] = A.pop()
            
            y_1 = X[j][file_1]
            y_2 = X[j][file_2]
            
            epsilon_1 = min(y_1, 1 - y_2)
            epsilon_2 = min(y_2, 1 - y_1)
            
            
            #first we will calculate phi_alpha and then phi_beta
            # Transform the X into alpha
            X[j][file_1] -= epsilon_1
            X[j][file_2] += epsilon_1
            
            phi_alpha = phi(Adj, X, Theta)
           
            
            #Transform the X into beta
            X[j][file_1] += epsilon_1 + epsilon_2
            X[j][file_2] -= epsilon_1 + epsilon_2
            
            phi_beta = phi(Adj, X, Theta)
            
            # if _phi_beta >= phi_alpha then we dont need to make any changes to X
            if(phi_beta < phi_alpha): 
                X[j][file_1] -= epsilon_1 + epsilon_2
                X[j][file_2] += epsilon_1 + epsilon_2
            
            to_round -= 2
            
            if(X[j][file_2] <0.999999 and X[j][file_2] > 0.00001): # error term and check if that file is non-intergal or not
                A.append([j,file_2])
                to_round += 1
                
            if(X[j][file_1] <0.99999 and X[j][file_1] > 0.00001): # error term and check if that file is non-intergal or not
                A.append([j,file_1])
                to_round += 1
            
            
        else: # that is the last (cache, file) pair is the only non-zero entry for the cache:
            # making it integral, by rounding it to the nearest integer
            [j, file_1] = A.pop()
            
            if(1 - X[j][file_1] < C - cache_fill[j]): # there is extra space left for this file to be cached
                cache_fill += 1 - X[j][file_1] #the extra fill
                X[j][file_1] = 1
            else:
                X[j][file_1] = 0

            to_round -= 1
    
    return X