import numpy as np
from numba import njit

"""
Generate synthetic data for IPLC using markov chain. 
For each user, initialize a random N x N transition matrix, where N is the number of files (library size) and store the generated sequence of states. Essentially, the requests made by any two users will be mutually independent.
"""
def generate_data(num_files:int, total_time:int)->np.ndarray:
    transitionMatrix:np.ndarray = np.random.rand(num_files,num_files) # generate random matrix
    transitionMatrix = transitionMatrix/transitionMatrix.sum(axis=1,keepdims=True) # normalize the rows
    state:int=np.random.randint(num_files)
    seq:np.ndarray=np.zeros(total_time)
    for i in range(total_time):
        state=np.random.choice(num_files,p=transitionMatrix[state,:])
        seq[i]=state
        if i%10000==0:
            print(i/100,end=" ",flush=True)
    return seq

def main():
    N=100
    T=1000000
    users=10
    for user in range(users):
        seq=generate_data(N,T)
        with open('data/synthetic_user_'+str(user)+'.npy','wb') as f:
            np.save(f,seq)

if __name__=="__main__":
    main()