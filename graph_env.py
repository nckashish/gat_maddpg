import numpy as np
from numpy import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Environment():
    def __init__(self, num_agents):
        self.n = num_agents


def adj_matrix():
    adj = [[0,1,0,1,0,1,1,1,0,0]
           [1,1,0,0,1,0,0,1,0,1]
           [0,1,0,1,0,1,1,1,0,0]
           [1,1,0,0,1,0,0,1,0,1]
           [0,1,0,1,0,1,1,1,0,0]
           [1,1,0,0,1,0,0,1,0,1]
           [0,1,0,1,0,1,1,1,0,0]
           [1,1,0,0,1,0,0,1,0,1]
           [0,1,0,1,0,1,1,1,0,0]
           [1,1,0,0,1,0,0,1,0,1]]
    
    return adj

def features():
    # 3 features values for each node.
    fet = [[0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]
           [0.1, 0.3, 0.5]]
    return fet


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def step(action_n):
    obs_n = features()
    rew_n = np.random.rand(1,10)
    #change done with actuial vlaues, 1 for every x episodes, 
    # look into max_episode_len.
    done_n =[0 for i in range(10)]

    return obs_n, rew_n, done_n

def action_space():
    a1 = random.rand() 
    a2 = random.rand() 
    return tf.Tensor([[a1, a2]], shape=(1, 2), dtype=float64)

