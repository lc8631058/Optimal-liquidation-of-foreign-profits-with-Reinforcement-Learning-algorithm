import numpy as np
import random
from collections import namedtuple, deque

from model import *
import math
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cuda:0"

class Agent():
    def __init__(self, state_size, action_size,
                 fc1_units, fc2_units, fc3_units, fc4_units, fc5_units,
                 LR, momentum, buffer_size, batch_size, update_every, tau,
                 eps_start,eps_end, eps_decay, seed):
        """
        :param state_size: dimension of state
        :param action_size: action space size
        :param fc1_units: FC layer 1 size
        :param fc2_units: FC layer 2 size
        :param fc3_units: FC layer 3 size
        :param LR: learning rate
        :param buffer_size: replay buffer capacity
        :param batch_size: batch size
        :param update_every: update the parameters of target network every update_every steps
        :param tau: the weight when updating target network's parameters
        :param eps_start: start value of epsilon greedy
        :param eps_end: end value of epsilon greedy
        :param eps_decay: decay rate of epsilon greedy
        :param seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size =  batch_size
        self.update_every = update_every
        # soft_update
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_policy = \
            QNetwork(state_size, action_size, seed,
                     fc1_units = fc1_units, fc2_units = fc2_units, fc3_units=fc3_units,
                     fc4_units = fc4_units, fc5_units = fc5_units).to(device)
        self.qnetwork_target = \
            QNetwork(state_size, action_size, seed,
                     fc1_units = fc1_units, fc2_units = fc2_units, fc3_units=fc3_units,
                     fc4_units=fc4_units, fc5_units=fc5_units).to(device)
        # copy parameters of policy network to target network
        self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())
        # set target to evaluation mode, not trainable
        self.qnetwork_target.eval()

        # self.optimizer = optim.RMSprop(self.qnetwork_policy.parameters(), lr=LR, momentum=momentum)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        # step for updating eps_threshold
        self.steps_done = 0

    def test(self, policy_state_dict, target_state_dict, optimizer_state_dict):
        self.qnetwork_policy.load_state_dict(policy_state_dict)
        self.qnetwork_target.load_state_dict(target_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.qnetwork_policy.eval()
        self.qnetwork_target.eval()

    def act(self, state, ifTest=False):
        """
        Returns actions for given state as per current policy
        :param state: current state
        :param eps: epsilon-greedy action selection
        :return: action
        """
        state = torch.from_numpy(state).float().to(device)

        # approaching eps_start, start from eps_start, end at eps_end
        # random > eps_threshold: follow policy
        # random <=  eps_threshold: random select action
        if ifTest:
            eps_threshold = self.eps_end
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1

        # Epsilon-greedy action selection
        if random.random() > eps_threshold:
            # set the network to evaluation module
            # self.qnetwork_policy.eval()

            with torch.no_grad():
                action_values = self.qnetwork_policy(state)
            # Set the network to training module
            # self.qnetwork_policy.train()
            return np.argmax(action_values.cpu().data.numpy(), axis=1).reshape([-1, 1])
        else:
            # action_size is 11, so the action space is [0, 1,..., 10]
            return np.array([random.choice(np.arange(self.action_size)) for _ in range(list(state.size())[0])]).reshape([-1, 1])

    def step(self, state, action, reward, next_state):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= 100 * self.batch_size and self.t_step % 4 == 0:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Learn every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # ----------- update the target network ----------- #
            self.soft_update(self.qnetwork_policy, self.qnetwork_target, self.tau)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples
        :param experiences: tuple of (s, a, r, s') tuples [torch.Variable]
        """
        states, actions, rewards, next_states = experiences

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s[0] != -1.,
                                                next_states)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([s.unsqueeze(0) for s in next_states
                                           if s[0] != -1.])

        # state_batch = torch.cat(states)
        # action_batch = torch.cat(actions)
        # reward_batch = torch.cat(rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.qnetwork_policy(states).gather(1, actions.long( ))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.qnetwork_target(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = next_state_values.unsqueeze(1) + rewards

        # Compute Huber loss
        # add one dimension to 1 position
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        # zero_grad: Clears the gradients of all optimized torch.Tensor
        self.optimizer.zero_grad()
        # Computes the gradient of current tensor
        loss.backward()
        # This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for self. The attribute will then contain the gradients computed and future calls to backward() will accumulate (add) gradients into it
        for param in self.qnetwork_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: as prediction model to learn
        :param target_model: produces fix target for the prediction model to calculate the loss
        :param tau: the portion to update the target parameters from local parameters
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        :param buffer_size: size of buffer
        :param batch_size: batch size
        :param seed: random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        """Add new experience to memory"""
        for STATE, ACTION, REWARD, NEXT_STATE in zip(state, action, reward, next_state):
            self.memory.append(Transition(STATE, ACTION, REWARD, NEXT_STATE))
        # abounden
        #     e = {"state": STATE, "action": ACTION, "reward": REWARD, "next_state": NEXT_STATE}
        #     # e = self.experience(state, action, reward, next_state, done)
        #     if not all(e) in self.memory:
        #         self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # This way can not filter the next_state that is None
        # # Creates a Tensor from a numpy.ndarray
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(np.vstack([e['done'] for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states)

    def __len__(self): 
        return len(self.memory)
