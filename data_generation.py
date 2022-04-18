import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import Dict,Tuple, Optional
"""
Generate synthetic data for IPLC using markov chain. 
For each user, initialize a random N x N transition matrix, where N is the number of files (library size) and store the generated sequence of states. Essentially, the requests made by any two users will be mutually independent.
"""
def generate_data(num_files:int, total_time:int)->np.ndarray:
    transitionMatrix:np.ndarray = np.random.rand(num_files,num_files) # generate random matrix
    mask=random_graph(num_files)
    transitionMatrix=transitionMatrix*mask
    transitionMatrix = transitionMatrix/transitionMatrix.sum(axis=1,keepdims=True) # normalize the rows
    state:int=np.random.randint(num_files)
    seq:np.ndarray=np.zeros(total_time)
    for i in tqdm(range(total_time)):
        state=np.random.choice(num_files,p=transitionMatrix[state,:])
        seq[i]=state
    return seq

# generates uniformly random (not really!) connected graph 
def random_graph(nodes:int=100,p:float=0.05)->np.ndarray:
    G=nx.gnp_random_graph(nodes,0.05)
    while(not nx.is_connected(G)):
        G=nx.gnp_random_graph(nodes,0.05)

    A=nx.adjacency_matrix(G)
    # A.to_numpy_array()
    return np.array(A.todense())

def main():
    N=50
    C=5
    S=50
    T=1000000
    p=np.array([0.5,0.25,0.125,0.0625,0.0625])
    seq=generate_data_fsm(N,C,S,T,p)
    np.save(f'synthetic_fsm_num_files{N}_cache_size{C}_S{S}_T{T}.npy',seq)
    # for user in tqdm(range(users)):
    #     seq=generate_data(N,T)
    #     np.save('synthetic_erdos_renyi_'+str(user+1)+'.npy',seq)

# Generate a random sequence of file requests for one user that can be deterministically predicted by an FSP with cache size C.
# First, initialize a random FSM and fix C files randomly for each state.
# Then at each timestep, output a random file from the list of C files and go to the next state.
def generate_data_fsm(
    N:int=10, # total number of files (alphabet size)
    C:int=5, # cache size
    S:int=10, # number of states
    T:int=100, # size of sequence
    pmf:Optional[np.ndarray]=None # pmf of size C 
    )->np.ndarray:

    g:np.ndarray = np.random.randint(S,size=(S,N))
    f:Dict[int,np.ndarray]= {}
    for i in range(S):
        f[i]=np.random.permutation(N)[:C]
    seq:np.ndarray=np.zeros(T,dtype=int)

    current_state=np.random.randint(S) #initial state
    for i in tqdm(range(T)):
        # x=f[current_state][np.random.randint(C)]
        if pmf is None:
            pfm=np.ones(C)/C

        x=np.random.choice(f[current_state],p=pmf)
        seq[i]=x
        current_state=g[current_state,x]
    print(seq.dtype)
    return seq

if __name__=="__main__":
    main()