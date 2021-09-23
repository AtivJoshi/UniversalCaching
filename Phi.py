import numpy as np
# to calculate the number of hits Adj is the adjacency matrix and X is the cache configuration
# A is of shape I,J where were as X is of shape J, F
# where I is the number of users and J is the number of caches and F is the number of files.

def phi(Adj, X, Theta):
        
        f_total_hits = 0 

        I, J = np.shape(Adj)
        J, F = np.shape(X)
        
        for i in range(I): # going through every user
            for f in range(F): # going through every file for that user
                
                #check if there is a cache with this file f connect to user i 
                f_hit = 1  #fractional hit 
                for j in np.flatnonzero(Adj[i][:]):  # going through all the caches connected to this user
        
                    f_hit *= (1- X[j][f])

                
                f_total_hits += Theta[i][f]*(1-f_hit) 
            
        return f_total_hits