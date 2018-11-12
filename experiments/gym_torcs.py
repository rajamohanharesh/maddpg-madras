import gym
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import numpy as np
import copy
import collections as col
import os
import time
import random
import threading
import queue


class TorcsEnv:
    terminal_judge_start = 250      # If after 100 timestep still no progress, terminated
    termination_limit_progress = 1  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50
    
    initial_reset = False

    obs_dim = 29
    act_dim = 3

    def __init__(self,vision=False, throttle=False, gear_change=False,num_agents = 1):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.n_collision = 0
        self.initial_run = True
        self.time_step = 0
        self.worker_threads = []
        self.currState = None 
        self.lock = threading.RLock()
        self.reset_lock = threading.RLock()
        self.queue = queue.Queue()
        self.reset_queue = queue.Queue()
        self.n = num_agents
        self.action_space = []
        self.observation_space = []

        # Now the action_space and observation_space are actually being used, just like in OpenAI's gym 
        if throttle is False:                           # Throttle is generally True
            for i in range(self.n):
                self.action_space.append(spaces.Box(low=-1.0, high=1.0, shape=(1,)))

        else:
            high = np.array([1., 1., 1.])
            low = np.array([-1., 0., 0.])
            for i in range(self.n):
                self.action_space.append(spaces.Box(low=low, high=high))              # steer, accel, brake (according to agent_to_torcs() (check the function definition))

        if vision is False:                             # Vision is True when you need the images from the simulator 
            high = np.inf*np.ones(self.obs_dim)
            low = -high
            for i in range(self.n):
                self.observation_space.append(spaces.Box(low, high))
        	# just like in https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L50 (as of 30/5/17)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255], dtype=theano.config.floatX)
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0], dtype=theano.config.floatX)
            for i in range(self.n):    
                self.observation_space.append(spaces.Box(low=low, high=high))			

    def terminate(self):
        episode_terminate = True
        client.R.d['meta'] = True
        print('Terminating because bad episode')

    def dummy(self,step,clients, actions, early_stop):
    	self.queue.put([step,clients,actions,early_stop])

    def reset_multi(self, clients, is_training,relaunch = False):
    	with self.reset_queue.mutex:
    		self.reset_queue.queue.clear()
    	
    	worker_threads = []
    	
    	obs_n = []
    	reset_clients = []
    	print("***************Clients rock and they are-*********",clients)
    	for i in range(self.n):
                		          
            t = threading.Thread(target = self.reset, args = (clients[i],is_training,relaunch))
            # worker_work = lambda: (self.step(step,clients[i],actions[i],early_stop))
            worker_threads.append(t)

    	for thread in worker_threads:
            thread.start()
    	
    	for thread in worker_threads:
            thread.join()

    	output_list = list(self.reset_queue.queue)
        
    	for q in output_list:
            obs_n.append(q[0])
            reset_clients.append(q[1])
     
    	return obs_n,reset_clients
    
    def step_torcs(self, step, clients, actions, early_stop):
    	with self.queue.mutex:
    		self.queue.queue.clear()
            
    	worker_threads = []
    	
    	obs_n = []
    	rew_n = []
    	done_n = []
    	info_n = []

    	for i in range(self.n):
            # print(step[i],clients[i],actions[i])                		          
            # worker_work = lambda: (self.step(step,clients[i],actions[i],early_stop))
            t = threading.Thread(target = self.step, args = (step,clients[i],actions[i],early_stop,i,clients))
            worker_threads.append(t)

    	for thread in worker_threads:
            thread.start()
    	
    	for thread in worker_threads:
            thread.join()

    	output_list = list(self.queue.queue)
    	for q in output_list:
            obs_n.append(q[0])
            rew_n.append(q[1])
            done_n.append(q[2])
            info_n.append(q[3])

    	return obs_n,rew_n,done_n,info_n
    
    def step(self, step, client, u, early_stop,index,clients):
        # client = self.client
        # lock = threading.RLock()
        # lock.acquire()
        this_action = self.agent_to_torcs(u)
        info = {'termination_cause':0}
        # Apply Action
        action_torcs = client.R.d
        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the previous full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        code = client.get_servers_input(step)

        if code==-1:
            client.R.d['meta'] = True
            print('Terminating because server stopped responding')
            self.queue.put([None, 0, client.R.d['meta'], {'termination_cause':'hardReset'}])
            return None, 0, client.R.d['meta'], {'termination_cause':'hardReset'}

        # Get the current full-observation from torcs
        obs = client.S.d
        # if index==0:
        #     obs1=clients[1].S.d
        #     sp1 = np.array(obs1['speedX'])
        # else:
        #     obs1=clients[0].S.d
        #     sp1 = np.array(obs1['speedX'])
        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)
        self.currState = np.hstack((self.observation.angle, self.observation.track, self.observation.trackPos, 
                                    self.observation.speedX, self.observation.speedY,  self.observation.speedZ, 
                                    self.observation.wheelSpinVel/100.0, self.observation.rpm))

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        # Added from Deepdriving
        dist_MM = np.array(obs['dist_MM'])
        dist_LL = np.array(obs['dist_LL'])
        dist_RR = np.array(obs['dist_RR'])
        dist_R = np.array(obs['dist_R'])
        dist_L = np.array(obs['dist_L'])
        dist_R_rear = np.array(obs['dist_R_rear'])
        dist_L_rear = np.array(obs['dist_L_rear'])

        # print("----------------------Distance to nearest cars--------------------------")
        # print(dist_LL,
        # dist_RR,
        # dist_R,
        # dist_L,
        # dist_R_rear,
        # dist_L_rear)
 
        
        # progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        
        reward = progress

                # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -300
            print('***collision***')

        # # Penalising if too near an opponent 
        if (dist_MM<8 and dist_MM>0) or (dist_LL<0.5 and dist_LL>0) or (dist_RR<0.5 and dist_RR>0) or \
            (dist_L<5 and dist_L>0) or (dist_R<5 and dist_R>0) or \
            (dist_L_rear<10 and dist_L_rear>0) or (dist_R_rear<10 and dist_R_rear>0):
            print('***Proximity***')
            reward = -300 
            if obs['damage'] - obs_pre['damage'] > 0:
                reward = -400
                self.n_collision = self.n_collision + 1
                print('***collision***')


        # if index==0:
        #     if sp*np.cos(obs['angle'])<sp1*np.cos(obs1['angle']):
        #         reward=reward-50
        #     elif (sp*np.cos(obs['angle'])-sp1*np.cos(obs1['angle']))>10:
        #         reward=reward-50

        #     if np.abs(obs['trackPos']-obs1['trackPos'])>5:
        #         reward-=50

        # if index==1:
        #     if sp*np.cos(obs['angle'])>sp1*np.cos(obs1['angle']):
        #         reward=reward-50
        #     elif (sp1*np.cos(obs1['angle'])-sp*np.cos(obs['angle']))>10:
        #         reward=reward-50
        #     if np.abs(obs['trackPos']-obs1['trackPos'])>4:
        #         reward-=50
        
        # if sp*np.cos(obs['angle'])<7:
        #     reward=reward-100
                # Termination judgement #########################
        episode_terminate = False
        # """
        if ( (abs(track.any()) > 1 or abs(trackPos) > 1)):  # Episode is terminated if the car is out of track
            reward = -2000
            episode_terminate = True
            client.R.d['meta'] = True
            info['termination_cause'] = 1
            print('Terminating because Out of Track')
        # """
        # temp_t_step=self.time_step-200
        # if self.terminal_judge_start < temp_t_step: # Episode terminates if the progress of agent is small
        #     if ( (progress < self.termination_limit_progress) and early_stop ):
        #         print("No progress")
        #         episode_terminate = True
        #         client.R.d['meta'] = True
        #         info['termination_cause'] = 2

        #         print('Terminating because Small Progress')
        # """
        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            reward=-2000
            episode_terminate = True
            client.R.d['meta'] = True
            info['termination_cause'] = 3
            print('Terminating because Turned Back')


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        # ob = self.observation
        # obs = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        self.queue.put([self.currState,reward, client.R.d['meta'], info])
        # lock.release()

        return self.observation, reward, client.R.d['meta'], info
        # return reward

    def reset(self, client,is_training, relaunch=False):
        #print("Reset")
        
        # reset_lock = threading.RLock()
        # reset_lock.acquire()
        port = client.port
        self.time_step = 0
        self.n_collision = 0

        print('11111111111111111111111111111')
        if self.initial_reset is not True:
            client.R.d['meta'] = True
            client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
            	self.reset_torcs(is_training)
            	print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        client = snakeoil3.Client(p=port, vision=self.vision)  # Open new UDP in vtorcs
        client.MAX_STEPS = np.inf

        # client = self.client
        client.get_servers_input(-1)  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs)
        self.currState = np.hstack((self.observation.angle, self.observation.track, self.observation.trackPos, 
                                    self.observation.speedX, self.observation.speedY,  self.observation.speedZ, 
                                    self.observation.wheelSpinVel/100.0, self.observation.rpm))

        self.last_u = None

        self.initial_reset = False
        self.reset_queue.put([self.currState, client])
        # reset_lock.release()
        return self.get_obs(), client

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self,is_training):
       #print("relaunch torcs")   
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            # os.system('torcs -r ~/.torcs/config/raceman/practice.xml -nofuel -nolaptime -vision &')
            os.system('torcs -r ~/.torcs/config/raceman/quickrace.xml -nofuel -nolaptime -vision &')
            #os.system('torcs -nofuel -nolaptime -vision &')
        else:
            # os.system('torcs -r ~/.torcs/config/raceman/practice.xml -nofuel -nolaptime -vision &')
            if is_training :
                os.system('torcs -r ~/.torcs/config/raceman/quickrace.xml -nofuel -nolaptime -vision &')
            else:
                os.system('torcs -nofuel -nolaptime &')
        if is_training == 0:
            time.sleep(0.5)
            print("------------------Going to autostart---------------------------")
            os.system('sh scripts/autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled             # This is generally true
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled        # This is generally false
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def make_observation(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel',
                     'dist_MM',
                     'dist_LL',
                     'dist_RR',
                     'dist_R',
                     'dist_L',
                     'dist_R_rear',
                     'dist_L_rear'
]
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               dist_L = np.array(raw_obs['dist_L'], dtype=np.float32)/60.,
                               dist_R = np.array(raw_obs['dist_R'], dtype=np.float32)/60.,
                               dist_L_rear = np.array(raw_obs['dist_L_rear'], dtype=np.float32)/30.,
                               dist_R_rear = np.array(raw_obs['dist_R_rear'], dtype=np.float32)/30.,
                               dist_LL = np.array(raw_obs['dist_LL'], dtype=np.float32)/60.,
                               dist_MM = np.array(raw_obs['dist_MM'], dtype=np.float32)/60.,
                               dist_RR = np.array(raw_obs['dist_RR'], dtype=np.float32)/60.
                               )
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)

def test_thread():
	env = TorcsEnv(vision=False, throttle=True, gear_change=False)
	num_workers = 3
	step = []
	clients = []
	actions = []

	for i in range(num_workers):
		step.append(i)
		clients.append(i)
		actions.append(i)

	env.step_torcs(step,clients,actions,0,num_workers)

if __name__ == '__main__':
	test_thread()

	
