import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import snakeoil3_gym as snakeoil3

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--port", type=int, default=3001, help="Port to be used for first car")
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=1400, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=600, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--early-stop", type=int, default= 0 , help="early stop value")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=350, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=30, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=350, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input            
        out = layers.fully_connected(out, num_outputs=num_units-50, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units+50, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def mlp_actor_model(input, scope, reuse=False, num_units=350, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        layer2_size = num_units+50
        initializer = tf.random_uniform_initializer(-1e-4,1e-4)
        out = input            
        out = layers.fully_connected(out, num_outputs=num_units-50, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=layer2_size, activation_fn=tf.nn.relu)
        
        # W_steer = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4))
        W_steer = tf.get_variable("W_steer",shape=[layer2_size,1],initializer=initializer)
        b_steer = tf.get_variable("b_steer",shape=[1],initializer=initializer)
        # b_steer = tf.Variable(tf.random_uniform([1],-1e-4,1e-4))

        # W_accel = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4))
        # b_accel = tf.Variable(tf.random_uniform([1],-1e-4,1e-4))
        W_accel = tf.get_variable("W_accel",shape=[layer2_size,1],initializer=initializer)
        b_accel = tf.get_variable("b_accel",shape=[1],initializer=initializer)


        # W_brake = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4))
        # b_brake = tf.Variable(tf.random_uniform([1],-1e-4,1e-4))

        W_brake = tf.get_variable("W_brake",shape=[layer2_size,1],initializer=initializer)
        b_brake = tf.get_variable("b_brake",shape=[1],initializer=initializer)

        steer = tf.tanh(tf.matmul(out,W_steer) + b_steer)
        accel = tf.sigmoid(tf.matmul(out,W_accel) + b_accel)
        brake = tf.sigmoid(tf.matmul(out,W_brake) + b_brake)
        
        # action_output = tf.concat(1, [steer, accel, brake])
        action_output = tf.concat([steer, accel, brake], 1)
        print("W,b",W_brake,b_brake)
        # out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return action_output

def make_env(scenario_name, arglist, benchmark=False):
    from gym_torcs import TorcsEnv

    env = TorcsEnv(vision=False, throttle=True, gear_change=False,num_agents =1)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    p_model = mlp_actor_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, p_model,model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i,p_model, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        clients = []
        port = arglist.port
        for i in range(env.n):
        
            clients.append(snakeoil3.Client(p=port+i, vision=False))
        obs_n = []
        for i in range(env.n):
        # Open new UDP in vtorcs
            clients[i].MAX_STEPS = np.inf
            clients[i].get_servers_input(0)  # Get the initial input from torcs

            obs = clients[i].S.d  # Get the current full-observation from torcs
            print("Client",i)
            ob = env.make_observation(obs)
            s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            obs_n.append(s)
            # print(ob)


        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        print('--------------------------------------DEBUG 1 --------------------------------------')
        # obs_n,clients = env.reset_multi(clients,False)
        episode_step = 0
        train_step = 0
        t_start = time.time()
        is_training = 1
        print('Starting iterations...')
        episode_no=0

        epsilon = 0.5
        epsilon_decay = 600000.0
        epsilon_steady_state = 0.01

        while True:
            # actions_n = []
            relaunch=True
            # get action
            # action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # for action in action_n:
            #     action[0] = np.clip(action[0],-1,1)
            #     action[1] = np.clip(action[1],0,1)
            #     action[2] = np.clip(action[2],0,1)
            #     actions_n.append(action)
            # default_action = np.array([0.,0.8,0.1])
            actions_n = [agent.noise_action(obs,epsilon) for agent, obs in zip(trainers,obs_n)] 
            # actions_n = [default_action for obs in obs_n]
            epsilon-=1/epsilon_decay

            epsilon = max(epsilon,epsilon_steady_state)

            print(epsilon)
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step_torcs(episode_step,clients,actions_n,arglist.early_stop)
            print('Action:',actions_n)
            episode_step += 1
            done = any(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for info in info_n:
                if info['termination_cause']=='hardReset':
                    relaunch = False 
            
            if relaunch:
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], actions_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew
            obs_n = new_obs_n

            if done or terminal:
                episode_no+=1
                if episode_no%30==0:
                    is_training=0
                obs_n, clients = env.reset_multi(clients,is_training,relaunch)
                is_training=1
                relaunch = True
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(fname=arglist.save_dir, saver=saver,time_step=episode_no)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
