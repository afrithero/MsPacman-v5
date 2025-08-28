import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
from strategies import TD_TARGETS
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # initialize env
        self.env = gym.make(config["env_id"])

        # initialize test_env
        self.test_env = gym.make(config["env_id"], render_mode = "rgb_array")
        self.test_env = gym.wrappers.RecordVideo(self.test_env, 'video')
        self.test_env.seed(0)

        # initialize behavior network and target network
        dueling = str(self.model_type).lower() == "dueling"
        self.behavior_net = AtariNetDQN(num_actions=self.env.action_space.n, dueling=dueling)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(num_actions=self.env.action_space.n, dueling=dueling)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

        # decide strategies
        self.td_target_fn = TD_TARGETS.get(self.algo)

    
    def decide_agent_actions(self, observation, epislon=0.0, action_space=None):
        if random.random() < epislon:
            action = np.random.randint(0, action_space.n)
        else:
            actions = self.behavior_net(observation)
            action = torch.argmax(actions).item()
        
        return action

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        # calculate the loss and update the behavior network
        q_esti = self.behavior_net(state)
        q_esti = q_esti.gather(1, action.to(torch.int64)) 
        
        q_target = self.td_target_fn(self.batch_size, self.behavior_net, self.target_net, next_state, reward, done, self.gamma)

        criterion = nn.MSELoss()
        loss = criterion(q_esti, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

