import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque

from agent import Agent


def initialize_env(unity_file):
    # Initialize the environment
    env = UnityEnvironment(file_name=unity_file, worker_id=2)

    # Get default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get state and action spaces
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    n_agents = len(env_info.agents)
    
    print('State size: ', state_size)
    print('Action size: ', action_size)
    print('Number of agents: ', n_agents)
    
    return env, brain_name, state_size, action_size, n_agents





def ddpg(env, brain_name,
         agent, n_agents,
         n_episodes=2000, t_max=3000):
    """Deep Determinitic Policy Gradient.

    Params
    ======
        env: unity environment object
        brain_name (string): brain name of initialized environment
        agent: initialized agent object
        n_episodes (int): maximum number of training episodes
        t_max (int): maximum timesteps in episode
    """
    
    scores = []
    scores_window = deque(maxlen=100)
    for e in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(n_agents)
        #for _ in range(1, t_max):
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if np.any(done):
                break

        # Max score
        max_score = np.max(score)
        scores_window.append(max_score)
        scores.append(max_score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(e, np.mean(scores_window), max_score), end="")
        if e % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
        if np.mean(scores_window)>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

def apply(env, brain_name, 
          agent, filepath_actor, 
          filepath_critic):
    load_checkpoints(agent, filepath_actor, filepath_critic)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state, add_noise=False)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print('Score: {}'.format(score))
    
def plot_scores(scores_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for key, scores in scores_dict.items():
        scores_smoothed = gaussian_filter1d(scores, sigma=5)
        plt.plot(np.arange(len(scores)), scores_smoothed, label=key)
    plt.ylabel('smoothed Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

def load_checkpoints(agent, filepath_actor, filepath_critic):
    agent.actor_local.\
        load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.\
        load_state_dict(torch.load('checkpoint_critic.pth'))

if __name__ == '__main__':
    # Hyperparameters
    N = 10000
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 512
    GAMMA = .99
    TAU = 1e-3
    LEARNING_RATE_ACTOR = 5e-4
    LEARNING_RATE_CRITIC = 5e-3
    WEIGHT_DECAY = 0.0
    UPDATE_LOCAL = 10
    N_UPDATES = 5
    SEED = 2
    
    env, brain_name, state_size, action_size, n_agents = \
        initialize_env('Tennis_Linux/Tennis.x86_64')

    # Initialize agent
    agent = Agent(state_size, action_size,
                  n_agents, buffer_size=BUFFER_SIZE, 
                  batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
                  lr_a=LEARNING_RATE_ACTOR, lr_c=LEARNING_RATE_CRITIC,
                  weight_decay=WEIGHT_DECAY, update_local=UPDATE_LOCAL,
                  n_updates=N_UPDATES, random_seed=SEED)
    
    # Train agent
    scores = ddpg(env, brain_name, agent, n_agents, n_episodes=N)
    
    plot_scores({'DDPG': scores})
    
    # Watching a smart agent
    apply(env, brain_name, 
          agent, 'checkpoint_actor.pth', 
          'checkpoint_critic.pth')
    
    env.close()