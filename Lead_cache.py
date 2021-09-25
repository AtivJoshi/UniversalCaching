import numpy as np
from numpy import linalg as LNG
from LPsolver import SolveLP
from rounding import cache_by_cache_pipage_rounding
from rounding_madow import madow_rounding
#import rounding_madow
from Phi import phi
import math
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

def Lead_cache(cache_request, Adj, T, F, C, d):
    
    # hit_rate = []
    # download_rate = []
    hit_rate_Madow = []
    download_rate_Madow = []
    
    I, J = np.shape(Adj)
    # sanity_check = True # verify if every caching configuration was admissible 
    
    #gamma has to be of the form I*F
    # we can just sample gamma once this does not change the upper bound gurarantees 
    gamma = np.random.normal(0, 1, (I, F))
    
    eta_constant = math.pow(I, 0.75)/(math.pow(2*d*(math.log(F/C) + 1), 0.25)*(math.pow(2*J*C, 0.5)))
    constr_violation_tol = 1.0
    #eta_constant = 0
    
    Xr = np.zeros((I,F)) # this accounts for the cumulative file request 
    
    for t in range(T):
        #print(t)
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0) # taking the non-negative part of theta only
        
        #start variable saves the solution to the LP in the last time step, and use it to initalise the LPsolver in this timestep.
        # just initialize start to zero?
        if t>0:
            (Y_f, start) = SolveLP(Adj, theta, C, start, t) # solves the LP and gets a fractional solution 
        else:
            (Y_f, start) = SolveLP(Adj, theta, C, 0, t) # dummy initializaton for t=0
        

        
        if(np.any(np.sum(Y_f,axis = 1) > C + constr_violation_tol)):# checks if all the caches has less than or equal to C files
           
            print("fractional solution trying to cache more than capacity")
            
        # lets do this for small library size, bigger library size might give us zero
        # Skipping pipage rounding if the LP solution is almost integral
        
        #tol=I*F*0.0001
        #if LNG.norm(Y_f-np.rint(Y_f), 'fro')< tol:
         #   Y = np.rint(Y_f)
        #else:

        #if(sanity_check == False):
    #    print("trying to cache more than capacity")

        # Running the Madow's sampling scheme to obtain a feasible solution

        Y_madow = np.rint(madow_rounding(Y_f, theta, Adj, C)) #rounding
        #calculating download_rate
        
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
            # sanity_check = False
            
        hits = 0
        
        for user in range(I): # goes through every user
            present = 0 
            for cache in np.flatnonzero(Adj[user][:]): # goes through every cache the user is connected to
                if Y_madow[cache][int(cache_request[user][t])] > 0.9998: #checks if the requested file at time t is present in the cache configuration and we also deal with precision error
                    present = 1
                    break
            hits += present
            
           
        hit_rate_Madow.append(hits) 

        # if(sanity_check == False):
        #     print("trying to cache more than capacity")

        # # calculating the madow first as it wont change Y_f
        # # there was a bug when we did cache_by_cache_pipage_rounding first as it would change the state of Y_f

        # Y = np.rint(cache_by_cache_pipage_rounding(Y_f, theta, Adj, C)) # rounding
        # #calculating download_rate
        
        # if(t> 0):
        #     difference = Y - Y_prev
        #     download = np.sum(difference[difference > 0]) 
        #     download_rate.append(download)
        #     Y_prev = Y
        # else:
        #     Y_prev = Y
        # #verify if the cache configurations are valid
        # if(np.any(np.sum(Y,axis = 1) > C+ constr_violation_tol)): # checks if all the caches has less than or equal to C files
           
        #     print("trying to cache more than capacity")
        #     sanity_check = False
            
        #print('cache configuration',Y)
        #updating Xr
        for i in range(I):
            m = cache_request[i][t]
            Xr[i][m] += 1
        #print('\n cumulative file request', Xr)
            
        # calculates the number of hits
        # hits = 0
        
        # for user in range(I): # goes through every user
        #     present = 0 
        #     for cache in np.flatnonzero(Adj[user][:]): # goes through every cache the user is connected to
        #         if Y[cache][int(cache_request[user][t])] > 0.9998: #checks if the requested file at time t is present in the cache configuration and we also deal with precision error
        #             present = 1
        #             break
        #     hits += present
            
           
        # hit_rate.append(hits) 

        if t%1000==0:
            print(f't: {t}\no. hits: {np.sum(hit_rate_Madow)}\nfrac hits: {(np.sum(hit_rate_Madow)/t):0.2f}\n\n') 
 
        #print(Y_f < 0)
        #print("Xr = {} \n Theta = {} \n Y_f = {} \nY_madow = {}\n Y = {} \n ----------------------------- \n".format(Xr, theta, Y_f, Y_madow, Y))
        
    return hit_rate_Madow, download_rate_Madow
