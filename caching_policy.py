import math
from tqdm import tqdm
import numpy as np

from LPsolver import SolveLP
from rounding_madow import madow_rounding

def iplc(input_seq:np.ndarray,adj_mat:np.ndarray, total_time:int,
        num_files:int, cache_size:int,deg:int)->int:
    """
        Incremental Parsing with Lead Cache
        
        input_seq: input sequence of file requests
        adj_mat: adjacency matrix of the bipartite graph
        total_time: algorithm will run for min(total_time, len(input_seq))
        num_files: library size
        cache_size: storage space of each cache machine
        deg: degree of each cache machine
    """

    num_users,num_cache=np.shape(adj_mat)

    gamma=np.random.normal(0,1,(num_users, num_files))
    eta_constant = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    constr_violation_tol = 1.0

    states={'epsilon':np.zeros((num_users,num_cache))}
    current_state='epsilon'
    num_hits=0

    for t in tqdm(range(total_time)):
        Xr=states[current_state]
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0) # taking the non-negative part of theta only

        # y_t: cache configuration predicted at time t
        y_t = SolveLP(adj_mat, theta, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))

        # update the cumulative request of current_state
        for i in range (num_users):
            m=input_seq[t,i]
            Xr[i,m]+=1


        for user in range(num_users):
            for f in np.flatnonzero(adj_mat[user][:]):
                if y_madow[f][int(input_seq[user][t])] >0.998:
                    num_hits+=1
                    break # since each user requests exactly one file

        # go to next state
        current_state=f'{current_state}:{np.array2string(input_seq[t,:])}'

        if current_state not in states:
            states[current_state]=np.zeros((num_users,num_cache))
            current_state='epsilon'
    return num_hits
    