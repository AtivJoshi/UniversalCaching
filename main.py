import pandas as pd
import sys
import time
import pytz
import os.path
import numpy as np
from datetime import datetime as dt
IST=pytz.timezone('Asia/Kolkata')

from caching_policy import *
from data_loader import *
from network_generator import *


def experiment1():
    # initialize parameters
    num_users=15
    num_caches=7
    deg=8
    num_files=300
    cache_size=int(0.1*num_files)
    total_time=9000
    folder_path=''

    g_path=folder_path+ f'data/graph_u{num_users}_c{num_caches}_d{deg}.npy'

    if os.path.isfile(g_path):
        graph=np.load(g_path)
        print('graph loaded')
    else:
        graph=generate_network_graph(num_users,num_caches,deg)
        np.save(folder_path+f'data/graph_u{num_users}_c{num_caches}_d{deg}.npy',graph)
        print('graph generated')

    # Load data
    dataset='cmu'
    if dataset=='cmu':
        data=load_cmu_data(num_users,num_files,folder_path=folder_path)
    elif dataset=='ratings':
        data=load_ratings_data(num_users,num_files,folder_path)
    else:
        data=np.array([0])
    data.shape

    # leadcache
    # cumulative_req2,hits2=lc(data,graph,total_time,num_files,cache_size,deg)
    # hitrate_leadcache=hits2/(cumulative_req2*num_users)
    hitrate_leadcache=0
    print(hitrate_leadcache)
    
    # Run iplc algorithm

    cumulative_req,hits,s=iplc_multiple_fsm(data,graph,total_time,num_files,cache_size,deg)

    # Make dataframe
    df=pd.DataFrame(list(cumulative_req.items()),columns=['state','visits'])
    df['hits']=df['state'].map(hits)
    df[['fsm','state']]=pd.DataFrame(df['state'].tolist(),index=df.index)
    df=df.loc[:,['fsm','state','visits','hits']]
    df.set_index(['fsm','state'],inplace=True,drop=True)
    df['hitrate']=df['hits']/df['visits']
    tstr=dt.now(IST).strftime('%m_%d_%H_%M_%S')

    # metadata
    data_description=f'Data is modified so the file names are changed to [1..len(total_unique_ids)]'

    # store in .h5 format 
    with pd.HDFStore(folder_path+f'data/results/gumbel_{dataset}_iplc_multifsm_u{num_users}_c{num_caches}_t{min(total_time,data.shape[0])}_d{deg}_f{num_files}_{tstr}.h5') as storage:
        storage['df']=df
        hitrate_iplc=df['hits'].sum()/df['visits'].sum()
        storage.get_storer('df').attrs.metadata={'users':num_users, 'caches':num_caches, 'number of files':num_files,
                                                'cache size':cache_size, 'time':min(total_time,data.shape[0]), 'dataset': dataset,
                                                'network_graph':graph, 'algo':'iplc_multiple_fsm', 'hitrate_leadcache':hitrate_leadcache,
                                                    'hitrate_iplc':hitrate_iplc, 'data_description': data_description}
        hitrate_iplc
        print()
        df1=df.drop(list(df[df['visits']<100].index))
        df1.sort_values(by=['fsm','hitrate'],inplace=True,ascending=[True,False])
        df1['hits'].sum()/df1['visits'].sum()
        df1

def main():
    # experiment1()
    data=np.load('data/synthetic_fsm_num_files50_cache_size5_S50_T1000000.npy')
    data=data[:,np.newaxis]
    data=data.astype('int64')
    cumulative_req,hits=hedge_single_cache(data,1000,50,5)
    

if __name__=='__main__':
    main()