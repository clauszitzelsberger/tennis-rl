import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    
    def __init__(self, state_size,
                 action_size, n_agents=1,
                 buffer_size=int(1e7), batch_size=256, 
                 gamma=.99, tau=1e-3, 
                 lr_a=1e-4, lr_c=1e-3, 
                 weight_decay=0, update_local=10,
                 n_updates=5, random_seed=1):
        
        """Initialize an Agent object
        
        Params
        =====
            state_size (int): Dimension of states
            action_size (int): Dimension of actions
            n_agents (int): Number of agents
            buffer_size (int): size of replay buffer
            batch_size (int): size of sample
            gamma (float): discount factor
            tau (float): (soft) update of target parameters
            lr_a (float): learning rate of actor
            lr_c (float): learning rate of critic
            weight_decay (float): L2 weight decay
            update_local (int): update local network every x steps
            n_updates (int): number of updates
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_agents = n_agents
        
        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.weight_decay = weight_decay
        self.update_local = update_local
        self.n_updates = n_updates
        
        # Actor networks
        self.actor_local = \
            [Actor(state_size, action_size, seed=random_seed).to(device) for _ in range(n_agents)]
        self.actor_target = \
            [Actor(state_size, action_size, seed=random_seed).to(device) for _ in range(n_agents)]
        self.actor_optimizer = \
            [optim.Adam(self.actor_local[i].parameters(), lr=lr_a) for i in range(n_agents)]
            
        # Critic networks
        self.critic_local = \
            [Critic(state_size*n_agents, action_size*n_agents, seed=random_seed).to(device) for _ in range(n_agents)]
        self.critic_target = \
            [Critic(state_size*n_agents, action_size*n_agents, seed=random_seed).to(device) for _ in range(n_agents)]
        self.critic_optimizer = \
            [optim.Adam(self.critic_local[i].parameters(), lr=lr_c, weight_decay=weight_decay) for i in range(n_agents)]
            
        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        # Time step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        #for state, action, reward, next_state, done in zip(state, action, reward, next_state, done):
        # Store states, actions, rewards and next states of all agents in one tuple
        self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            if self.t_step % self.update_local == 0:
                for i in range(self.n_agents):
                    for _ in range(self.n_updates):
                        sample = self.memory.sample()
                        self.__learn(sample, self.gamma, agent=i)
        

    def act(self, state, add_noise=True):
        """Returns action given a state according to current policy

        Params
        ======
            state (array_like): current state
            add_noise (bool): handles exploration
        """
        actions = []
        state = torch.from_numpy(state).float().to(device)
        for i in range(self.n_agents):
            self.actor_local[i].eval()
            with torch.no_grad():
                action = self.actor_local[i](state).cpu().data.numpy()
            self.actor_local[i].train()
    
            if add_noise:
                action += self.noise.sample()
            actions.append(action[i])
        return np.clip(actions, -1, 1)

    
    def reset(self):
        self.noise.reset()
    
    def __learn(self, sample, gamma, agent):
        """
        Params
        ======
            sample (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            agent (int): agent id
        """
        states, actions, rewards, next_states, dones = sample
        
        
        observations = torch.\
            from_numpy(states.swapaxes(0,1)).\
            float().to(device)
        
        next_observations = torch.\
            from_numpy(next_states.swapaxes(0,1)).\
            float().to(device)
        
        states = torch.\
            from_numpy(states.reshape(self.batch_size, -1)).\
            float().to(device)
            
        next_states = torch.\
            from_numpy(next_states.reshape(self.batch_size, -1)).\
            float().to(device)
            
        actions = torch.\
            from_numpy(actions.reshape(self.batch_size, -1)).\
            float().to(device)
        
        rewards = torch.\
            from_numpy(np.expand_dims(np.take(rewards, agent, axis=1), axis=1)).\
            float().to(device)
        
        
        #----------------- Critic
        # Next actions and actions values
        actions_next = \
            np.array([self.actor_target[i](next_observations[i]).cpu().data.numpy() 
                      for i in range(self.n_agents)])
    
        actions_next = torch.\
            from_numpy(actions_next.swapaxes(0,1).reshape(self.batch_size, -1)).\
            float().to(device)
        
        Q_targets_next = self.critic_target[agent](next_states, actions_next)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local Critic network
        Q_expected = self.critic_local[agent](states, actions)
        
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize loss
        self.critic_optimizer[agent].zero_grad()
        critic_loss.backward()
        self.critic_optimizer[agent].step()
        
        #----------------- Actor
        # Compute actor loss
        actions_pred = \
            np.array([self.actor_local[i](observations[i]).cpu().data.numpy() 
                      for i in range(self.n_agents)])
    
        actions_pred = torch.\
            from_numpy(actions_pred.swapaxes(0,1).reshape(self.batch_size, -1)).\
            float().to(device)
            
        self.actor_local[agent](observations[agent])
        actor_loss = -self.critic_local[agent](states, actions_pred).mean()
        
        # Minimize loss
        self.actor_optimizer[agent].zero_grad()
        actor_loss.backward()
        self.actor_optimizer[agent].step()
        
        #----------------- update target networks
        self.__soft_update(self.critic_local[agent], self.critic_target[agent], self.tau)
        self.__soft_update(self.actor_local[agent], self.actor_target[agent], self.tau)
        
    
    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param \
            in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.\
                copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
             
        
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples in"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batchparamteres
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = \
            namedtuple('Experience',
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.\
            append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        #dones = np.array([int(e.done) for e in experiences if e is not None])
        dones = torch.\
            from_numpy(np.vstack([e.done for e in experiences if e is not None]).\
            astype(np.uint8)).float().to(device)
        #dones = torch.\
        #    from_numpy(np.array([int(e.done) for e in experiences if e is not None])).\
        #    float().to(device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
 
           
class OUNoise:
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=.0, theta=.15, sigma=.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """Reset the internal state (=noise) to mean (mu)"""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + \
            self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state