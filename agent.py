import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
n_agents = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    
    def __init__(self, state_size,
                 action_size,
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
            [Actor(state_size, action_size, seed=random_seed*i).to(device) for i in range(n_agents)]
        self.actor_target = \
            [Actor(state_size, action_size, seed=random_seed*i).to(device) for i in range(n_agents)]
        self.actor_optimizer = \
            [optim.Adam(self.actor_local[i].parameters(), lr=lr_a) for i in range(n_agents)]
            
        # Critic networks
        self.critic_local = \
            [Critic((state_size + action_size)*n_agents, action_size, seed=random_seed*i).to(device) for i in range(n_agents)]
        self.critic_target = \
            [Critic((state_size + action_size)*n_agents, action_size, seed=random_seed*i).to(device) for i in range(n_agents)]
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
        
        state1 = state[0]
        state2 = state[1]
        
        action1 = action[0]
        action2 = action[1]
        
        reward1 = reward[0]
        reward2 = reward[1]
        
        next_state1 = next_state[0]
        next_state2 = next_state[1]
        
        done1 = done[0]
        done2 = done[1]
        
        
        self.memory.add(state1, state2, 
                        action1, action2, 
                        reward1, reward2,
                        next_state1, next_state2,
                        done1, done2)

        self.t_step += 1
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            if self.t_step % self.update_local == 0:
                for i in range(n_agents):
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
        for i in range(n_agents):
            observation = \
                torch.from_numpy(np.array([state[i]])).float().to(device)

            self.actor_local[i].eval()
            with torch.no_grad():
                action = self.actor_local[i](observation).cpu().data.numpy()
            self.actor_local[i].train()
    
            if add_noise:
                action += self.noise.sample()
            actions.append(action[0])
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
        
        s1, s2, a1, a2, r1, r2, ns1, ns2, d1, d2 = sample
        
        # Next actions and actions values
        na1 = self.actor_target[0](ns1)
        na2 = self.actor_target[1](ns2)
        
        # Predicted actions
        pred_a1 = self.actor_local[0](s1)
        pred_a2 = self.actor_local[1](s2)
        
        if agent == 0:
            states = [s1, s2]
            actions = [a1, a2]
            next_actions = [na1, na2]
            pred_actions = [pred_a1, pred_a2]
            next_states = [ns1, ns2]
            rewards = [r1, r2]
            dones = [d1, d2]
        elif agent==1:
            states = [s2, s1]
            actions = [a2, a1]
            next_actions = [na2, na1]
            pred_actions = [pred_a2, pred_a1]
            next_states = [ns2, ns1]
            rewards = [r2, r1]
            dones = [d2, d1]
        else:
            raise 'Wrong number of agents'
        
        #----------------- Critic
        Q_targets_next = \
            self.critic_target[agent](next_states[0], next_states[1], 
                                      next_actions[0], next_actions[1])
        
        # Compute Q targets for current states
        Q_targets = rewards[agent] + (gamma * Q_targets_next * (1 - dones[agent]))

        # Get expected Q values from local Critic network
        Q_expected = self.critic_local[agent](states[0], states[1], 
                                              pred_actions[0], pred_actions[1])
        
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize loss
        self.critic_optimizer[agent].zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer[agent].step()
        
        #----------------- Actor        
        actions[0] = pred_actions[0]
            
        # Compute loss
        actor_loss = -self.critic_local[agent](states[0], states[1], 
                                               actions[0], actions[1]).mean()
        
        # Minimize loss
        self.actor_optimizer[agent].zero_grad()
        actor_loss.backward(retain_graph=True)
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
                       field_names=['state1', 'state2',
                                    'action1', 'action2',
                                    'reward1', 'reward2',
                                    'next_state1', 'next_state2',
                                    'done1', 'done2'])
        self.seed = random.seed(seed)

    def add(self, state1, state2, 
            action1, action2, 
            reward1, reward2, 
            next_state1, next_state2, 
            done1, done2):
        """Add a new experience to memory"""
        self.memory.\
            append(self.experience(state1, state2,
                                   action1, action2,
                                   reward1, reward2,
                                   next_state1, next_state2,
                                   done1, done2))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        state1 = torch.\
            from_numpy(np.vstack([e.state1 for e in experiences if e is not None])).\
            float().to(device)
        state2 = torch.\
            from_numpy(np.vstack([e.state2 for e in experiences if e is not None])).\
            float().to(device)
        
        action1 = torch.\
            from_numpy(np.vstack([e.action1 for e in experiences if e is not None])).\
            float().to(device)
        action2 = torch.\
            from_numpy(np.vstack([e.action2 for e in experiences if e is not None])).\
            float().to(device)
        
        reward1 = torch.\
            from_numpy(np.vstack([e.reward1 for e in experiences if e is not None])).\
            float().to(device)
        reward2 = torch.\
            from_numpy(np.vstack([e.reward2 for e in experiences if e is not None])).\
            float().to(device)
        
        next_state1 = torch.\
            from_numpy(np.vstack([e.next_state1 for e in experiences if e is not None])).\
            float().to(device)
        next_state2 = torch.\
            from_numpy(np.vstack([e.next_state2 for e in experiences if e is not None])).\
            float().to(device)
        
        done1 = torch.\
            from_numpy(np.vstack([e.done1 for e in experiences if e is not None]).astype(np.uint8)).\
            float().to(device)
        done2 = torch.\
            from_numpy(np.vstack([e.done2 for e in experiences if e is not None]).astype(np.uint8)).\
            float().to(device)

        return (state1, state2, 
                action1, action2, 
                reward1, reward2, 
                next_state1, next_state2, 
                done1, done2)



    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
 
           
class OUNoise:
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=.0, theta=.5, sigma=.8):
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