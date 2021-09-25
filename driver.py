import math

import numpy as np
import pandas as pd

from Generate_network import generate_network_graph
from IPLC import iplc

# For single user
users =1
time_limit = 1000000
caches = 1
alpha = 0.2
d = 1
# Dropping all file requests with id larger than the threshold to reduce the library size
threshold = 300 # maximum number of files (N) 
NumSeq = 3  # denotes the number of non-overlapping sequences over which the experminent is run

# generates a random network
Adj = generate_network_graph(users, caches, d)

# Generating the request sequence
#rating dataset
# data = pd.read_csv("ratings1m.dat", sep = '::', engine='python')
# data.columns = ['User_ID', 'File_ID', 'Ratings', 'Timestamp']
# DataLength = len(data)

# cmu dataset
RowLimit=150000
data = pd.read_csv("CMU_huge.txt", sep = ' ')
# data = data[1:RowLimit]
data.columns = ['Req_ID', 'File_ID', 'File_Size']
DataLength = len(data)
offset = 1000
horizon = 1000*users

# splitting up the entire time axis into non-overlapping parts
for i in range(NumSeq):

    # rating dataset
    df =data[int(i*DataLength/NumSeq) : int((i+1)*DataLength/NumSeq)]
    # df.sort_values("Timestamp")

    # cmu dataset
    # df = pd.DataFrame(data[i*offset : i*offset+ horizon])

    
    # Renaming the annoynimized FileID's
    old_id = df.File_ID.unique()
    old_id.sort()
    new_id = dict(zip(old_id, np.arange(len(old_id))))
    df = df.replace({"File_ID": new_id})
    # df.sort_values("Timestamp") # not needed for cmu dataset 
     
    # Reducing the library size
    df = df[df.File_ID < threshold]
    df = df.reset_index(drop=True) 
    library_size = df['File_ID'].max()+2
    C = math.floor(alpha*library_size) # cache size 
    v = df['File_ID']
    RawSeq = np.array(v)
    time = int(np.floor(min(time_limit, len(v)/(users))))-1 # N
    print(f'T: {time}, {v.shape}')
    # RawSeq contains an array of requests
    # df = np.array_split(RawSeq, users) #no need for one user
    # df=np.array(df)
    df=np.array(RawSeq)
    df=df[np.newaxis,:]
    hit_rates_Madow, download_rates_Madow = iplc(df, Adj, time, library_size, C, d)
    print("hit rate: ", hit_rates_Madow )
