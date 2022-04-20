import math
from typing import Tuple, Dict
from tqdm import tqdm
import numpy as np

from LPsolver import SolveLP
from rounding_madow import madow_rounding

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
    # gamma:np.ndarray=np.random.normal(0,1,(num_users,num_files))
    gamma:np.ndarray=np.random.gumbel(0,1,(num_users,num_files))
    eta_constant:float = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    constr_violation_tol = 1.0

    visits:int=0
    hits:int=0
    Xr:np.ndarray=np.zeros((num_users,num_files))
    for t in tqdm(range(total_time)):
        # taking the non-negative part of theta only
        # multiplying t with eta for eta_t as in step 5 of algorithm 1
        theta = np.maximum(Xr + eta_constant*(math.pow((t+1),0.5))*gamma, 0)  

        # y_t: cache configuration predicted at time t
        y_t,_ = SolveLP(adj_mat, theta, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))


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

    return visits, hits

def hedge_single_cache(
        input_seq:np.ndarray, 
        # adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int
        # deg:int
    )->Tuple[Dict[str,int],Dict[str,int]]:
    """Hedge for single cache setting
        
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

    # num_users, num_cache=np.shape(adj_mat)
    total_time=min(input_seq.shape[0],total_time)
    
    # constants as defined in lead cache algorithm
    gamma:np.ndarray=np.random.normal(0,1,(1,num_files))
    # gamma:np.ndarray=np.random.gumbel(0,1,(num_users,num_files))
    # eta_constant:float = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    eta_constant:float = 1/(math.pow(2*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*cache_size, 0.5)))
    # constr_violation_tol = 1.0

    # A state is labeled by the parse sequence of LZ algorithm, where the initial 
    # state is labeled 'epsilon'.
    # State corresponding to sequence (34,45,90) is labeled by 'epsilon:34:45:90'
    init_state='epsilon'

    # dict that stores cumulative request (no. of times a file is requested)
    # at a given state for each user. Instead of nested dictionary, using dictionary with tuple as a key
    # index is (user,current_state)
    # user_states_cumulative_request:Dict[Tuple[int,str],np.ndarray]={}
    user_states_cumulative_request:Dict[str,np.ndarray]={}

    # current state of a given user
    # current_state:Dict[int,str]={}

    # no of times a state is visited for a particular user
    # states_visits:Dict[Tuple[int,str],int]={}
    states_visits:Dict[str,int]={}

    # no of hits at a given state of a user
    # states_hits:Dict[Tuple[int,str],int]={}
    states_hits:Dict[str,int]={}

    # initialize 
    # for i in range(num_users):
    #     user_states_cumulative_request[(i,init_state)]=np.zeros(num_files) # index is (user,current_state)
    #     current_state[i]=init_state # index is user
    #     states_hits[(i,init_state)]=0 # index is (user,current_state)
    #     states_visits[(i,init_state)]=0 # index is (user,current_state)

    user_states_cumulative_request[init_state]=np.zeros(num_files)
    current_state=init_state
    states_hits[init_state]=0
    states_visits[init_state]=0

    for t in tqdm(range(total_time)):

        #use current_state of FSM of each user to build Xr and compute theta
        Xr=np.zeros((1,num_files))
        
        # array of how many times the current_state of a user is visited
        # time_array=np.zeros((num_users,1))
        time_array=0

        # Building Xr. For each user, take the array of cumulative request of the current state.
        # No. of visits of current_state is maintained separately for each user. This is used to build eta.
        # for i in range(num_users):
        #     Xr[i,:]=user_states_cumulative_request[(i,current_state[i])]
        #     time_array[i]=states_visits[(i,current_state[i])]+1
        Xr[0,:]=user_states_cumulative_request[current_state]
        time_array=states_visits[current_state]+1

        # Get theta by perturbing Xr using eta and gamma
        theta = np.maximum(Xr + eta_constant*(np.sqrt(time_array))*gamma, 0) # taking the non-negative part of theta only

        # y_t: cache configuration predicted at time t
        # y_t,_ = SolveLP(adj_mat, theta, cache_size, t)
        # y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))
        y_madow=np.argsort(theta)[0][::-1][:cache_size]
        # print(y_madow)
        # y_madow=y_madow[:2]
        # observe the incoming cache request at time t and update the cumulative request and total visit count of current_state for each user
        # for u in range(num_users):
        #     states_visits[(u,current_state[u])]+=1 
        #     m:int=input_seq[t,u] # file requested at by user u at time t
        #     user_states_cumulative_request[(u,current_state[u])][m]+=1
        states_visits[current_state]+=1 
        m:int=input_seq[t,0] # file requested at by user u at time t
        user_states_cumulative_request[current_state][m]+=1

        # update the total hit count of current state
        # for user in range(num_users):
        #     for f in np.flatnonzero(adj_mat[user][:]):
        #         if y_madow[f][int(input_seq[t,user])] >0.998:
        #             states_hits[(user,current_state[user])]+=1
        #             break # since a user requests exactly one file at a time
        if m in y_madow:
            states_hits[current_state]+=1

        # for each user go to next state
        # for user in range(num_users):
        #     next_state=f'{current_state[user]}:{np.array2string(input_seq[t,user])}'
        next_state=f'{current_state}:{np.array2string(input_seq[t,0])}'


            # if the next_state is encountered for the first time, then initialize it and go to init_state, else go to next_state
            # if (user,next_state) not in user_states_cumulative_request:
            #     user_states_cumulative_request[(user,next_state)]=np.zeros(num_files)
            #     states_hits[(user,next_state)]=0
            #     states_visits[(user,next_state)]=0
            #     current_state[user]=init_state
            # else:
            #     current_state[user]=next_state
        if next_state not in user_states_cumulative_request:
            user_states_cumulative_request[next_state]=np.zeros(num_files)
            states_hits[next_state]=0
            states_visits[next_state]=0
            current_state=init_state
        else:
            current_state=next_state

    return states_visits, states_hits

def iplc_multiple_fsm(
        input_seq:np.ndarray, 
        adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int,
        deg:int
    )->Tuple[Dict[Tuple[int,str],int],Dict[Tuple[int,str],int],Dict]:
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
    # gamma:np.ndarray=np.random.gumbel(0,1,(num_users,num_files))
    eta_constant:float = math.pow(num_users, 0.75)/(math.pow(2*deg*(math.log(num_files/cache_size) + 1), 0.25)*(math.pow(2*num_cache*cache_size, 0.5)))
    constr_violation_tol = 1.0

    # A state is labeled by the parse sequence of LZ algorithm, where the initial 
    # state is labeled 'epsilon'.
    # State corresponding to sequence (34,45,90) is labeled by 'epsilon:34:45:90'
    init_state='epsilon'

    # dict that stores cumulative request (no. of times a file is requested)
    # at a given state for each user. Instead of nested dictionary, using dictionary with tuple as a key
    user_states_cumulative_request:Dict[Tuple[int,str],np.ndarray]={}

    # current state of a given user
    current_state:Dict[int,str]={}

    # no of times a state is visited for a particular user
    states_visits:Dict[Tuple[int,str],int]={}

    # no of hits at a given state of a user
    states_hits:Dict[Tuple[int,str],int]={}

    # initialize 
    for i in range(num_users):
        user_states_cumulative_request[(i,init_state)]=np.zeros(num_files) # index is (user,current_state)
        current_state[i]=init_state # index is user
        states_hits[(i,init_state)]=0 # index is (user,current_state)
        states_visits[(i,init_state)]=0 # index is (user,current_state)

    for t in tqdm(range(total_time)):

        #use current_state of FSM of each user to build Xr and compute theta
        Xr=np.zeros((num_users,num_files))
        
        # array of how many times the current_state of a user is visited
        time_array=np.zeros((num_users,1))

        # Building Xr. For each user, take the array of cumulative request of the current state.
        # No. of visits of current_state is maintained separately for each user. This is used to build eta.
        for i in range(num_users):
            Xr[i,:]=user_states_cumulative_request[(i,current_state[i])]
            time_array[i]=states_visits[(i,current_state[i])]+1

        # Get theta by perturbing Xr using eta and gamma
        theta = np.maximum(Xr + eta_constant*(np.sqrt(time_array))*gamma, 0) # taking the non-negative part of theta only

        # y_t: cache configuration predicted at time t
        y_t,_ = SolveLP(adj_mat, theta, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, theta, adj_mat, cache_size))

        # observe the incoming cache request at time t and update the cumulative request and total visit count of current_state for each user
        for u in range(num_users):
            states_visits[(u,current_state[u])]+=1 
            m:int=input_seq[t,u] # file requested at by user u at time t
            user_states_cumulative_request[(u,current_state[u])][m]+=1

        # update the total hit count of current state
        num_hits=0
        for user in range(num_users):
            for f in np.flatnonzero(adj_mat[user][:]):
                if y_madow[f][int(input_seq[t,user])] >0.998:
                    states_hits[(user,current_state[user])]+=1
                    break # since a user requests exactly one file at a time

        # for each user go to next state
        for user in range(num_users):
            next_state=f'{current_state[user]}:{np.array2string(input_seq[t,user])}'

            # if the next_state is encountered for the first time, then initialize it and go to init_state, else go to next_state
            if (user,next_state) not in user_states_cumulative_request:
                user_states_cumulative_request[(user,next_state)]=np.zeros(num_files)
                states_hits[(user,next_state)]=0
                states_visits[(user,next_state)]=0
                current_state[user]=init_state
            else:
                current_state[user]=next_state
    return states_visits, states_hits, {}


def markov_offline(
        input_seq:np.ndarray, 
        adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int,
        deg:int,
        k:int
    ):

    '''
        k: order of markov fsm
    '''

    num_users, num_cache=np.shape(adj_mat)
    total_time=min(input_seq.shape[0],total_time)
    markov:Dict[Tuple[int,Tuple],np.ndarray]={}
    # current state of a given user
    current_state:Dict[int,Tuple]={}

    # no of times a state is visited for a particular user
    states_visits:Dict[Tuple[int,Tuple],int]={}

    # no of hits at a given state of a user
    states_hits:Dict[Tuple[int,Tuple],int]={}

    for i in range(num_users):
        markov[(i,tuple(input_seq[:k,i]))]=np.zeros(num_files) # index is (user,current_state)
        current_state[i]=tuple(input_seq[:k,i]) # index is user
        states_hits[(i,tuple(input_seq[:k,i]))]=0 # index is (user,current_state)
        states_visits[(i,tuple(input_seq[:k,i]))]=0 # index is (user,current_state)

    for t in range(k,total_time):
        for u in range(num_users):
            states_visits[(u,current_state[u])]+=1
            m:int=input_seq[t,u]
            markov[(u,current_state[u])][m]+=1
            next_state=current_state[u][1:]+(m,)
            if (u,next_state) not in markov:
                markov[(u,next_state)]=np.zeros(num_files)
                states_hits[(u,next_state)]=0
                states_visits[(u,next_state)]=0
            current_state[u]=next_state

    # for single cache, the optimal policy is to fetch most often requested files
    # for multiple caches, do we approximate using relaxed LP + madow's sampling?
    # no need to add random vector in offline algorithm
    for i in range(num_users):
        current_state[i]=tuple(input_seq[:k,i])

    for t in tqdm(range(k,total_time)):
        Xr=np.zeros((num_users,num_files))
        for i in range(num_users):
            # print(f'{t} {i} {current_state[i]}')
            Xr[i,:]=markov[(i,current_state[i])]
        y_t,_ = SolveLP(adj_mat, Xr, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, Xr, adj_mat, int(cache_size)))
        for u in range(num_users):
            for f in np.flatnonzero(adj_mat[u][:]):
                if y_madow[f][int(input_seq[t,u])] >0.998:
                    states_hits[(u,current_state[u])]+=1
                    break # since a user requests exactly one file at a time
        for u in range(num_users):
            m:int=input_seq[t,u]
            next_state=current_state[u][1:]+(m,)
            current_state[u]=next_state
        
    return states_visits,states_hits


def markov_online(
        input_seq:np.ndarray, 
        adj_mat:np.ndarray,
        total_time:int,
        num_files:int,
        cache_size:int,
        deg:int,
        k:int
    ):

    num_users, num_cache=np.shape(adj_mat)

    markov:Dict[Tuple[int,Tuple],np.ndarray]={}

    # current state of a given user
    current_state:Dict[int,Tuple]={}

    # no of times a state is visited for a particular user
    states_visits:Dict[Tuple[int,Tuple],int]={}

    # no of hits at a given state of a user
    states_hits:Dict[Tuple[int,Tuple],int]={}

    for i in range(num_users):
        markov[(i,tuple(input_seq[:k,i]))]=np.zeros(num_files) # index is (user,current_state)
        current_state[i]=tuple(input_seq[:k,i]) # index is user
        states_hits[(i,tuple(input_seq[:k,i]))]=0 # index is (user,current_state)
        states_visits[(i,tuple(input_seq[:k,i]))]=0 # index is (user,current_state)

    for t in tqdm(range(k,total_time)):
        Xr=np.zeros((num_users,num_files))
        for i in range(num_users):
            # print(f'{t} {i} {current_state[i]}')
            Xr[i,:]=markov[(i,current_state[i])]
        y_t,_ = SolveLP(adj_mat, Xr, cache_size, t)
        y_madow = np.rint(madow_rounding(y_t, Xr, adj_mat, int(cache_size)))

        # observe the incoming cache request at time t and update the cumulative request and total visit count of current_state for each user
        for u in range(num_users):
            states_visits[(u,current_state[u])]+=1
            m:int=input_seq[t,u]
            markov[(u,current_state[u])][m]+=1

        # update the total hit count of current state
        for u in range(num_users):
            for f in np.flatnonzero(adj_mat[u][:]):
                if y_madow[f][int(input_seq[t,u])] >0.998:
                    states_hits[(u,current_state[u])]+=1
                    break # since a user requests exactly one file at a time
        # for each user go to next state
        for u in range(num_users):
            m:int=input_seq[t,u]
            next_state=current_state[u][1:]+(m,)
            if (u,next_state) not in markov:
                markov[(u,next_state)]=np.zeros(num_files)
                states_hits[(u,next_state)]=0
                states_visits[(u,next_state)]=0
            current_state[u]=next_state
    return states_visits,states_hits

def binary_markov_offline(input_seq:np.ndarray,
        total_time:int,
        k:int
    ):
    markov:Dict[Tuple[int,...],np.ndarray]={tuple(input_seq[:k]):np.array([0,0])}
    
    current_state:Tuple=tuple(input_seq[:k])
    # states_visits or states_hits are not required
    # states_hits:Dict[Tuple,int]={tuple(input_seq[:k]):0}
    for t in tqdm(range(total_time)):
        markov[current_state][input_seq[t]]+=1
        # print(f't: {t}')
        # print(f'cs: {current_state}, inp: {input_seq[t]}')
        # print(markov)
        
        next_state=current_state[1:]+(input_seq[t],)
        if next_state not in markov:
            markov[next_state]=np.array([0,0])
        current_state=next_state
    hits=0
    for key in markov:
        hits+=markov[key].max()
    return markov,hits

def binary_markov_online(input_seq:np.ndarray,
        total_time:int,
        k:int
    ):
    markov:Dict[Tuple[int,...],np.ndarray]={tuple(input_seq[:k]):np.array([0,0])}
    current_state:Tuple=tuple(input_seq[:k])
    hits=0
    for t in tqdm(range(total_time)):
        prediction=np.uint8(markov[current_state].argmax())
        # print(f'curr: {current_state}')
        # print(f'markov: {markov}')
        # print(f'pred: {prediction}')
        # print(f'nb: {input_seq[t]}\n')
        if input_seq[t]==prediction:
            hits+=1
        markov[current_state][input_seq[t]]+=1
        next_state=current_state[1:]+(input_seq[t],)

        if next_state not in markov:
            markov[next_state]=np.array([0,0])
        current_state=next_state
    return markov,hits

def binary_ip(input_seq:np.ndarray,
        total_time:int
    ):
    fsm:Dict[Tuple[int,...],np.ndarray]={():np.array([0,0])}
    current_state:Tuple=tuple()
    hits=0
    for t in tqdm(range(total_time)):
        pred=np.uint8(fsm[current_state].argmax())
        # print(f'curr: {current_state}')
        # print(f'markov: {fsm}')
        # print(f'pred: {pred}')
        # print(f'nb: {input_seq[t]}\n')
        if input_seq[t]==pred:
            hits+=1
        fsm[current_state][input_seq[t]]+=1
        next_state=current_state+(input_seq[t],)

        if next_state not in fsm:
            fsm[next_state]=np.array([0,0])
            current_state=()
        else:
            current_state=next_state
    return fsm,hits


def main():
    arr=np.random.randint(2,size=(10),dtype=np.int8)
    print(arr)
    m,h=binary_ip(arr,arr.shape[0])
    print(h,m)

if __name__=='__main__':
    main()
