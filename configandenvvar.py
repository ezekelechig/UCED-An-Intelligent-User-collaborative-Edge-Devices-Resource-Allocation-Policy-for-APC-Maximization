import numpy as np
from helper import *
import ipdb as pdb
import tensorflow as tf

class MecTerm(object):
    
    
    def __init__(self, end_config, main_config):
        self.rate = end_config['rate']
        self.dis = end_config['dis']
        self.id = end_config['id']
        self.state_dim = end_config['state_dim']
        self.action_dim = end_config['action_dim']
        self.action_bound = end_config['action_bound']
        self.data_buf_size = end_config['data_buf_size']
        self.t_factor = end_config['t_factor']
        self.penalty = end_config['penalty']
        
        self.sigma2 = main_config['sigma2']
        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        if 'model' not in end_config:
            self.channelModel = MarkovModel(self.dis, seed=main_config['random_seed'])
        else:
            n_t = 1
            n_r = end_config['num_r']
            self.channelModel = ARModel(self.dis, n_t, n_r, seed=train_config['random_seed'])
        
        self.DataBuf = 0
        self.Channel = self.channelModel.getCh()
        self.SINR = 0
        self.Power = np.zeros(self.action_dim)
        self.Reward = 0
        self.State = []
        
        # some pre-defined parameters
        self.k = 1e-27
        self.t = 0.001
        self.L = 500
    
    def localProc(self, p):
        return np.power(p/self.k, 1.0/3.0)*self.t/self.L/1000
    
    def localProcRev(self, b):
        return np.power(b*1000*self.L/self.t, 3.0)*self.k
    
    def offloadRev(self, b):
        return (np.power(2.0, b)-1)*self.sigma2/np.power(np.linalg.norm(self.Channel),2)
    
    def offloadRev2(self, b):
        return self.action_bound if self.SINR <= 1e-12 else (np.power(2.0, b)-1)/self.SINR
    
    def getCh(self):
        return self.Channel
    
    def setSINR(self, sinr):
        self.SINR = sinr
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.State = np.array([self.DataBuf, sinr, channel_gain])
        
    def sampleData(self):
        data_t = np.log2(1 + self.Power[0]*self.SINR)
        data_p = self.localProc(self.Power[1])
        over_power = 0
        
        self.DataBuf -= data_t+data_p
        if self.DataBuf < 0:
            over_power = self.Power[1] - self.localProcRev(np.fmax(0, self.DataBuf+data_p))
            self.DataBuf = 0
            
        data_r = np.random.poisson(self.rate)
        self.DataBuf += data_r
        return data_t, data_p, data_r, over_power
    
    def sampleCh(self):
        self.Channel = self.channelModel.sampleCh()
        return self.Channel
    
    def reset(self, rate, seqCount):
        self.rate = rate
        self.DataBuf = np.random.randint(0, self.data_buf_size-1)/2.0
        self.sampleCh()
        
        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True
            
        return self.DataBuf
    
class MecTermLD(MecTerm):
    
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        
        saver = tf.train.import_meta_graph(user_config['meta_path'])
        saver.restore(sess, user_config['model_path'])
 
        graph = tf.get_default_graph()
        input_str = "input_" + self.id + "/X:0"
        output_str = "output_" + self.id + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)
        
    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = sinr
        
      #buffer update
        [data_t, data_p, data_r, over_power] = self.sampleData()
        
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf
        
        self.sampleCh()

        # actor and critic network uodate
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        next_state = np.array([self.DataBuf, sinr, channel_gain])

        # system state update
        self.State = next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power)-over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
    
    def predict(self, isRandom):
        self.Power = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        return self.Power, np.zeros(self.action_dim)
        
class MecTermDQN_LD(MecTermLD):
   
    def __init__(self, sess, user_config, train_config):
        MecTermLD.__init__(self, sess, user_config, train_config)
        graph = tf.get_default_graph()
        self.action_level = user_config['action_level']
        self.action = 0
        
        output_str = "output_" + self.id + "/BiasAdd:0"
        self.out = graph.get_tensor_by_name(output_str)
        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])
        
    def predict(self, isRandom):
        q_out = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        self.action = np.argmax(q_out)
        
        action_tmp = self.action
        for i in range(self.action_dim):
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        return self.Power, np.zeros(self.action_dim)
        
class MecTermGD(MecTerm):
    
    def __init__(self, user_config, train_config, policy):
        MecTerm.__init__(self, user_config, train_config)
        self.policy = policy #         
        self.local_proc_max_bits = self.localProc(self.action_bound) # max processed bits per slot
        
    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = sinr

        # data buffer update
        [data_t, data_p, data_r, over_power] = self.sampleData()
        
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf
        
    
        self.sampleCh()
        
        # actor and critic network update
        
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        next_state = np.array([self.DataBuf, sinr, channel_gain])

        # system state update
        self.State = next_state
        
        sum_power = np.sum(self.Power)-over_power
        return self.Reward, np.sum(self.Power), 0, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
    
    def predict(self, isRandom):
        data = self.DataBuf
        if self.policy == 'local':
            self.offloadDo(self.localProcDo(data))
        else: 
            self.localProcDo(self.offloadDo(data))
        
        self.Power = np.fmax(0, np.fmin(self.action_bound, self.Power))
        return self.Power, np.zeros([self.action_dim])
    
    def localProcDo(self, data):
        if self.local_proc_max_bits <= data:
            self.Power[1] = self.action_bound
            data -= self.local_proc_max_bits
        else:
            self.Power[1] = self.localProcRev(data)
            data = 0
        return data
    
    def offloadDo(self, data):
        offload_max_bits = np.log2(1+np.power(np.linalg.norm(self.Channel),2)*self.action_bound/self.sigma2)
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev(data)
            data = 0
        return data
    
class MecTermGD_M(MecTermGD):
    def offloadDo(self, data):
        offload_max_bits = np.log2(1+self.SINR*self.action_bound)
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev2(data)
            data = 0
        return data

class MecTermRL(MecTerm):
    
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, user_config, train_config)
        
        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False

    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = sinr

        # update the data buffer
        [data_t, data_p, data_r, over_power] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        next_state = np.array([self.DataBuf, sinr, channel_gain])
        
        self.agent.update(self.State, self.Power, self.Reward, done, next_state, self.isUpdateActor)

        # update system state
        self.State = next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power)-over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow

    def predict(self, isRandom):
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        self.Power = np.fmax(0, np.fmin(self.action_bound, power))
        
        return self.Power, noise

class MecTermDQN(MecTerm):
    
    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.action_level = user_config['action_level']
        self.agent = DQNAgent(sess, user_config, train_config)
        self.action = 0
        
        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])
        
    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = sinr

        # update the data buffer
        [data_t, data_p, data_r, over_power] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        next_state = np.array([self.DataBuf, sinr, channel_gain])
        self.agent.update(self.State, self.action, self.Reward, done, next_state)

        # update system state
        self.State = next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power)-over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow

    def predict(self, isRandom):
        self.action, noise = self.agent.predict(self.State)
        
        action_tmp = self.action
        for i in range(self.action_dim):
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        
        return self.Power, noise

class MecSvrEnv(object):

   # Simulation environment
    
    def __init__(self, user_list, num_att, sigma2, max_len): 
        self.user_list = user_list
        self.num_user = len(user_list)
        self.num_att = num_att
        self.sigma2 = sigma2
        self.count = 0
        self.seqCount = 0
        self.max_len = max_len
        
        # specially designed for Greedy agent training
#         self.data_set = []
        
    def init_target_network(self):
        for user in self.user_list:
            user.agent.init_target_network()

    def step_transmit(self, isRandom=True):
        # get the channel vectors 
        channels = np.transpose([user.getCh() for user in self.user_list])
        # get the transmit powers
        powers = []
        noises = []
        
        for i in range(self.num_user):
            p, n = self.user_list[i].predict(isRandom)
            powers.append(p.copy())
            noises.append(n.copy())
        # compute the sinr for each user
        
#         self.data_set.append([self.user_list[0].State, powers[0]])        
        
        powers = np.array(powers)
        noises = np.array(noises)
        sinr_list = self.compute_sinr(channels, powers[:,0])

        rewards = np.zeros(self.num_user)
        powers = np.zeros(self.num_user)
        over_powers = np.zeros(self.num_user)
        data_ts = np.zeros(self.num_user)
        data_ps = np.zeros(self.num_user)
        data_rs = np.zeros(self.num_user)
        data_buf_sizes = np.zeros(self.num_user)
        next_channels = np.zeros(self.num_user)
        isOverflows = np.zeros(self.num_user)
        
        self.count += 1
        # feedback the sinr to each user
        for i in range(self.num_user):
            [rewards[i], powers[i], over_powers[i], data_ts[i], data_ps[i], data_rs[i], data_buf_sizes[i], next_channels[i], isOverflows[i]] = self.user_list[i].feedback(sinr_list[i], self.count >= self.max_len)
            
        return rewards, self.count >= self.max_len, powers, over_powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows        

    def compute_sinr(self, channels, powers):

        # Spatial-Domain MU-MIMO ZF
        H_inv = np.linalg.pinv(channels)
        noise = np.power(np.linalg.norm(H_inv, axis=1),2)*self.sigma2
        sinr_list = 1/noise
        
        return sinr_list

    def reset(self, isTrain=True):
        self.count = 0
        
        if isTrain:
            init_data_buf_size = [user.reset(user.rate, self.seqCount) for user in self.user_list]
            # get the channel vectors   
            channels = np.transpose([user.getCh() for user in self.user_list])
            # get the transmit powers to start
            powers = [np.random.uniform(0, user.action_bound) for user in self.user_list]
            # compute the sinr for each user
            sinr_list = self.compute_sinr(channels, powers)
        else:
            init_data_buf_size = [0 for user in self.user_list]
            sinr_list = [0 for user in self.user_list]

        for i in range(self.num_user):
            self.user_list[i].setSINR(sinr_list[i])
            
        self.seqCount += 1
        return init_data_buf_size


