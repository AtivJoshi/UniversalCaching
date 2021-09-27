import pandas as pd

from caching_policy import *
from data_loader import *
from network_generator import *


def main():
    # initialize parameters
    num_users=1
    num_caches=1
    deg=1
    num_files=10
    cache_size=3
    total_time=100

    # load data and graphs
    data=load_cmu_data(num_users,num_files)
    graph=generate_network_graph(num_users,num_caches,deg)
    cumulative_req,hits=iplc(data,graph,total_time,num_files,cache_size,deg)

if __name__=='__main__':
    main()