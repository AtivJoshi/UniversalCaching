import numpy as np
import pandas as pd
import os.path

# split
def split_data(raw_seq:np.ndarray,num_users:int)->np.ndarray:
    num_requests=raw_seq.size//num_users
    input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_requests))
    return input_seq

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

def main():
    # print(load_ratings_data(1,100))
    print(load_cmu_data(3,100))
    # print(load_synthetic_data(1,100))

if __name__ == "__main__":
    main()