import numpy as np
import pandas as pd
import os.path
from tqdm import tqdm

# split
def split_data(raw_seq:np.ndarray,num_users:int)->np.ndarray:
    num_requests=raw_seq.size//num_users
    input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
    return input_seq.T

# sorting by timestamp
def load_ratings_data(num_users:int, num_files:int,folder_path:str="")->np.ndarray:
    file_name='ratings'
    file_path=folder_path+f'data/{file_name}.dat'
    cache_path=folder_path+f'data/{file_name}_{num_users}u_{num_files}f_cache.npy'

    if os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        df = pd.read_csv(file_path, sep = '::', engine='python')
        df.columns = ['User_ID', 'File_ID', 'Ratings', 'Timestamp']
        df.sort_values(by='Timestamp',inplace=True) 
        # Total number of files = 3706, total users=6040
        # To control the size of the library, we can rename the file i to (i % num_files). 
        # This results in extremely bad accuracy, so avoiding it. Instead, drop the files when file_name > num_files.
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, np.arange(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.drop(list(df[df['File_ID']>=num_files].index),inplace=True) ##pyright: reportGeneralTypeIssues=false

        # Array of file requests
        raw_seq=df['File_ID'].to_numpy()
        print(raw_seq[:20])
        print('shape ', raw_seq.shape)
        print('size ',raw_seq.size)

        # Split raw_seq into chunks of size <num_users>
        num_requests=raw_seq.size//num_users
        input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_requests))
        np.save(cache_path,input_seq)
        return input_seq

def load_cmu_data(num_users:int, num_files:int,folder_path:str="",file_name="CMU_huge")->np.ndarray:
    file_path=folder_path+f'data/{file_name}.txt'
    cache_path=folder_path+f'data/{file_name}_{num_users}u_{num_files}f_cache.npy'

    if os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        df = pd.read_csv(file_path, sep = ' ',engine='python')
        df.columns = ['Req_ID', 'File_ID', 'File_Size']
        # To control the size of the library, we can rename the file i to (i % num_files). 
        # This results in extremely bad accuracy, so avoiding it. Instead, drop the files when file_name > num_files.
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, np.arange(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.drop(list(df[df['File_ID']>=num_files].index),inplace=True) ##pyright: reportGeneralTypeIssues=false

        # array of file requests
        raw_seq=df['File_ID'].to_numpy()

        # split raw_seq into chunks of size <num_users>
        num_requests=raw_seq.size//num_users
        input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
        np.save(cache_path,input_seq.T)
        return input_seq.T 

def load_synthetic_data(num_users:int, num_files:int, idx:int=0, folder_path:str="")->np.ndarray:
    file_name=f'synthetic_user_{idx}.npy'
    file_path=folder_path+f'data/{file_name}'

    raw_seq=np.load(file_path)
    # split raw_seq into chunks of size <num_users>
    num_requests=raw_seq.size//num_users
    input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
    return input_seq.T.astype('int64')


def iter_chunk_by_id_ratings(path,chunk_size):
    '''
    reference: https://stackoverflow.com/a/58567203/6361960
    Ratings data is already sorted by userid. 
    We read all rows belonging to each userId, 
    sort them by timestamp, and yield the array of movieId.
    '''
    csv_reader=pd.read_csv(path,iterator=True,chunksize=chunk_size,header=0,error_bad_lines=False,parse_dates=['timestamp'])
    term=0
    chunk = pd.DataFrame()

    for l in csv_reader:
        hits=l['userId'].astype(float).diff().dropna().to_numpy().nonzero()[0]
        if not len(hits):
            # if all ids are same
            chunk = chunk.append(l[['userId','movieId','timestamp']])
        else:
            start = 0
            for i in range(len(hits)):
                new_id = hits[i]+1
                chunk = chunk.append(l[['userId','movieId','timestamp']].iloc[start:new_id, :])
                chunk.sort_values(by='timestamp',inplace=True)
                yield chunk['movieId'].to_numpy()
                chunk = pd.DataFrame()
                start = new_id
            chunk = l[['userId','movieId','timestamp']].iloc[start:, :]
    yield chunk['movieId'].to_numpy()

def ratings25m():
    arr=np.empty(shape=25000100,dtype=np.uint16)
    start=0
    path=f'data/ratings25m/ml-25m/ratings.csv'
    with tqdm(total=25000100) as pbar:
        for c in iter_chunk_by_id_ratings(path,500):
            arr[start:start+c.shape[0]]=c
            start+=c.shape[0]
            pbar.update(c.shape[0])
        np.save(file='data/ratings25m/ml-25m/ratings.npy',arr=arr[:25000095])

def main():
    # print('ratings: ', load_ratings_data(3,100))
    print(load_cmu_data(3,100))
    # print(load_synthetic_data(1,100))

if __name__ == "__main__":
    main()