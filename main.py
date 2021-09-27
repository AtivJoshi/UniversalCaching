#%%
from caching_policy import *
from data_loader import *
from network_generator import *
# %%

def main():
    num_users=1
    num_caches=1
    deg=1
    num_files=100
    cache_size=10
    total_time=100
    # %%
    data=load_cmu_data(num_users,num_files)
    data=data[:total_time]
    #%%
    graph=generate_network_graph(num_users,num_caches,deg)
    h,i=iplc(data,graph,total_time,num_files,cache_size,deg)

if __name__=='__maine__':
    main()