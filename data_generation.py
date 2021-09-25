import numpy as np
from numba import njit

"""
Generate synthetic data for IPLC using markov chain. 
For each user, initialize a random N x N transition matrix, where N is the number of files (library size) and store the generated sequence of states. Essentially, the requests made by any two users will be mutually independent.
"""
# @njit
def generate_data(N:int, T:int)->np.ndarray:
    transitionMatrix:np.ndarray = np.random.rand(N,N) # generate random matrix
    transitionMatrix = transitionMatrix/transitionMatrix.sum(axis=1,keepdims=True) # normalize the rows
    state:int=np.random.randint(N)
    seq:np.ndarray=np.zeros(T)
    for i in range(T):
        state=np.random.choice(N,p=transitionMatrix[state,:])
        seq[i]=state
        if i%10000==0:
            print(i," ")
    return seq

def main():
    N=100
    T=1000000
    users=10
    user=1
    for user in range(users)[user:]:
        seq=generate_data(N,T)
        with open('data/synthetic_user_'+str(user)+'.npy','wb') as f:
            np.save(f,seq)

if __name__=="__main__":
    main()