import math

import numpy as np
from numpy import linalg as LNG

from LPsolver import SolveLP
from Phi import phi
from rounding import cache_by_cache_pipage_rounding
from rounding_madow import madow_rounding

# the data will come in a table with variable name cache_request
# where every row gives us the information about which user wants to access which file at that time t
# cache_request is the data matrix of form T*I
# Adj is the adjacency matrix of the network is of the form I*J
# T is the time uptill you want to run
# F is the libtrary size
# We just need to return the number of hit rates
# C is the number of files per cache
# returns a list of hit_rate over time.
# This code implements both the deterministic Pipage rounding as well as the randomized Madow's sampling-based rounding

def iplc(cache_request, Adj, T, F, C, d):
    
    hit_rate_Madow = []
    download_rate_Madow = []
    start=0
    I, J = np.shape(Adj)
    
    #gamma has to be of the form I*F
    # we can just sample gamma once this does not change the upper bound gurarantees 
    gamma = np.random.normal(0, 1, (I, F))
    
    eta_constant = math.pow(I, 0.75)/(math.pow(2*d*(math.log(F/C) + 1), 0.25)*(math.pow(2*J*C, 0.5)))
    constr_violation_tol = 1.0

    # this accounts for the cumulative file request
    states={"epsilon":np.zeros((I,F))}
    s="epsilon"
    # Xr = np.zeros((I,F))   
    for t in range(T):
        Xr=states[s]
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0) # taking the non-negative part of theta only

        # start variable saves the solution to the LP in the last time step, 
        # and use it to initalise the LPsolver in this timestep.
        # solves the LP and gets a fractional solution 
        (Y_f, start) = SolveLP(Adj, theta, C, start, t) 
        # checks if all the caches has less than or equal to C files
        if(np.any(np.sum(Y_f,axis = 1) > C + constr_violation_tol)):
            print("fractional solution trying to cache more than capacity")
            
        Y_madow = np.rint(madow_rounding(Y_f, theta, Adj, C)) #rounding

        #calculating download_rate, aka number of fetches
        if(t> 0):
            difference = Y_madow - Y_prev_madow
            download = np.sum(difference[difference > 0]) 
            download_rate_Madow.append(download)
            Y_prev_madow = Y_madow
        else:
            Y_prev_madow = Y_madow

        #verify if the cache configurations are valid
        if(np.any(np.sum(Y_madow,axis = 1) > C+ constr_violation_tol)): # checks if all the caches has less than or equal to C files
            print("trying to cache more than capacity")
        
        #updating Xr
        for i in range(I):
            m = cache_request[i][t]
            Xr[i][m] += 1

        hits = 0
        
        for user in range(I): # goes through every user
            present = 0 
            for cache in np.flatnonzero(Adj[user][:]): # goes through every cache the user is connected to
                if Y_madow[cache][int(cache_request[user][t])] > 0.9998: #checks if the requested file at time t is present in the cache configuration and we also deal with precision error
                    present = 1
                    break
            hits += present
        hit_rate_Madow.append(hits)
        s=s+":"+np.array2string(cache_request[:,t])
        if s not in states:
            states[s]=np.zeros((I,F))
            s="epsilon"
        # print(t," ",s)
        if t%1000==0:
            print(f't: {t}\no. hits: {np.sum(hit_rate_Madow)}\nfrac hits: {(np.sum(hit_rate_Madow)/t):0.2f}\n\n') 
    return np.sum(hit_rate_Madow), download_rate_Madow
