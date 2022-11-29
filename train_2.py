import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import maddpg.common.tf_util as U
from models.gat_maddpg import MADDPGAgentTrainer
import tf_slim as layers

import graph_env as environment

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

#q-function model
def gat_mlp_model(input, adj, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        #GAT Layer Variables
        nonlinearity = tf.nn.elu
        hid_units = [8]
        ffd_drop = 0.0
        attn_drop = 0.0
        n_heads = [8, 1] # additional entry for the output layer

        # adj matrix to bias, input is features???
        bias_mat = environment.adj_to_bias(adj, input.shape[0], nhood=1)  
        # first layer of GAT
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(input, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=nonlinearity,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        out = tf.add_n(attns) / n_heads[-1]
        # send it to output of MADDPG
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

#p- function model 
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

# critic is gat_mlp model, actor is mlp model 
# clear up action space. 
def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model_critic = gat_mlp_model
    model_actor = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model_critic, model_actor, obs_shape_n, env.action_space(), i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model_critic, model_actor, obs_shape_n, env.action_space(), i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):

    with U.single_threaded_session():
        U.initialize()
        #make environment for gat_maddpg
        env = environment

        batch_size = 1
        nb_nodes = 10
        adj = env.adj_matrix()
        #observations are features (B,N,F), features [x,y,z]
        obs_n = env.features()

        num_adversaries = 0
        obs_shape_n = [obs_n[i].shape for i in range(nb_nodes)]
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        

        t_len = 200
        episode_step = 0
        
        for t_step in range(t_len):
            # get action space. 
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            new_obs_n, rew_n, done_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            if done or terminal:
                obs_n = env.features()
                episode_step = 0
            

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, t_step)




    


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)