import math
from typing import Tuple, Dict
from tqdm import tqdm
import numpy as np

from LPsolver import SolveLP
from rounding_madow import madow_rounding

def iplc(
        input_seq:np.ndarray, 
        adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int,
        deg:int
    )->Tuple[Dict[str,int],Dict[str,int]]:
    """Incremental Parsing with Lead Cache
        
        Arguments:
        input_seq: input sequence of file requests
        adj_mat: adjacency matrix of the bipartite graph
        total_time: algorithm will run for min(total_time, len(input_seq))
        num_files: library size
        cache_size: storage space of each cache machine
        deg: degree of each cache machine

        Outputs:
        states_visits: total number of time the state is visited
        states_hits: total number of hits at the state
    """

    num_users, num_cache=np.shape(adj_mat)
    total_time=min(input_seq.shape[0],total_time)
    # constants as defined in lead cache algorithm
    gamma:np.ndarray=np.random.normal(0,1,(num_users,num_files))
    eta_constant:float = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    constr_violation_tol = 1.0

    # A state is labeled by the parse sequence of LZ algorithm, where the initial 
    # state is labeled 'epsilon'.
    # State corresponding to sequence (34,45,90) is labeled by 'epsilon:34:45:90'
    init_state='epsilon'

    # dict that stores cumulative request for each state
    states_cumulative_request={init_state:np.zeros((num_users,num_files))}

    # dict that store number of hits for each state
    states_hits={init_state:0}
    states_visits={init_state:0}
    current_state=init_state

    for t in tqdm(range(total_time)):
        Xr:np.ndarray=states_cumulative_request[current_state]
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0) # taking the non-negative part of theta only

        # y_t: cache configuration predicted at time t
        y_t,_ = SolveLP(adj_mat, theta, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))

        # update the cumulative request and total visit count of current_state
        states_visits[current_state]+=1
        for i in range (num_users):
            m=input_seq[t,i]
            Xr[i,m]+=1

        # update the total hit count of current state
        num_hits=0
        for user in range(num_users):
            for f in np.flatnonzero(adj_mat[user][:]):
                if y_madow[f][int(input_seq[t,user])] >0.998:
                    num_hits+=1
                    break # since each user requests exactly one file
        states_hits[current_state]+=num_hits

        # go to next state
        next_state=f'{current_state}:{np.array2string(input_seq[t,:])}'

        # if the next_state is encountered for the first time, then initialize it and go to init_state, else go to next_state
        if next_state not in states_cumulative_request:
            states_cumulative_request[next_state]=np.zeros((num_users,num_files))
            states_hits[next_state]=0
            states_visits[next_state]=0
            current_state=init_state
        else:
            current_state=next_state
    return states_visits, states_hits

def lc(
        input_seq:np.ndarray, 
        adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int,
        deg:int
    )->Tuple[int,int]:
    """Incremental Parsing with Lead Cache
        
        Arguments:
        input_seq: input sequence of file requests
        adj_mat: adjacency matrix of the bipartite graph
        total_time: algorithm will run for min(total_time, len(input_seq))
        num_files: library size
        cache_size: storage space of each cache machine
        deg: degree of each cache machine

        Outputs:
        states_visits: total number of time the state is visited
        states_hits: total number of hits at the state
    """

    num_users, num_cache=np.shape(adj_mat)
    total_time=min(input_seq.shape[0],total_time)
    # constants as defined in lead cache algorithm
    gamma:np.ndarray=np.random.normal(0,1,(num_users,num_files))
    eta_constant:float = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    constr_violation_tol = 1.0

    # A state is labeled by the parse sequence of LZ algorithm, where the initial 
    # state is labeled 'epsilon'.
    # State corresponding to sequence (34,45,90) is labeled by 'epsilon:34:45:90'
    # init_state='epsilon'

    # dict that stores cumulative request for each state
    # states_cumulative_request={init_state:np.zeros((num_users,num_files))}

    # dict that store number of hits for each state
    # states_hits={init_state:0}
    # states_visits={init_state:0}
    # current_state=init_state
    visits:int=0
    hits:int=0
    Xr:np.ndarray=np.zeros((num_users,num_files))
    for t in tqdm(range(total_time)):
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0) # taking the non-negative part of theta only

        # y_t: cache configuration predicted at time t
        y_t,_ = SolveLP(adj_mat, theta, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))

        # update the cumulative request and total visit count of current_state
        # states_visits[current_state]+=1
        visits+=1
        for i in range (num_users):
            m=input_seq[t,i]
            Xr[i,m]+=1

        # update the total hit count of current state
        for user in range(num_users):
            for f in np.flatnonzero(adj_mat[user][:]):
                if y_madow[f][int(input_seq[t,user])] >0.998:
                    hits+=1
                    break # since each user requests exactly one file

        # go to next state
        # next_state=f'{current_state}:{np.array2string(input_seq[t,:])}'

        # if the next_state is encountered for the first time, then initialize it and go to init_state, else go to next_state
        # if next_state not in states_cumulative_request:
        #     states_cumulative_request[next_state]=np.zeros((num_users,num_files))
        #     states_hits[next_state]=0
        #     states_visits[next_state]=0
        #     current_state=init_state
        # else:
        #     current_state=next_state
    return visits, hits
