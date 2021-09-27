import numpy as np
import random


def generate_network_graph(n, m, d): 
#where n is the number of users and m is the number of caches
	not_connected = True
	Graph = np.zeros((n,m))
	while(not_connected):

		Graph = np.zeros((n,m))
		for cache in range(m):
			adj_vertices = random.sample(range(n), d)
			Graph[adj_vertices,cache] = 1


		if(not np.any(np.sum(Graph, axis= 1) == 0)):
			not_connected=False
	return Graph