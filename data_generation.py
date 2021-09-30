import numpy as np
from tqdm import tqdm
import networkx as nx

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
    N=100
    T=1000000
    users=1
    for user in range(users):
        seq=generate_data(N,T)
        with open('data/synthetic_erdos_renyi_'+str(user+1)+'.npy','wb') as f:
            np.save(f,seq)

if __name__=="__main__":
    main()