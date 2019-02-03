# Tennis
Udacity's Deep Reinforcement Learning Nanodegree Project 'Tennis': Training agents
to play tennis.

## The Challenge
The challenge is to train two agents to control their rackets to bounce a ball over a net. 
A successfull hit over the net is rewarded with a positive value of +0.1. 
If the agent fails, e.g. if it lets the ball hit the ground or hits the ball out of bounds, it receives a negative reward of 0.01.
Therefore both agents' goal is to keep the ball in play.  

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation.  

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
Specifically,


- After each episode, the rewards that each agent received (without discounting) are added up, to get a score for each agent. This yields 2 (potentially different) scores, where the maximum of these 2 scores is considered.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name name python=3.6
	source activate name
	```
	- __Windows__: 
	```bash
	conda create --name name python=3.6 
	activate name
	```

2. Clone the repository and navigate to the folder.  Then, install the dependencies in the `requirements.txt` file.
```bash
git clone https://github.com/clauszitzelsberger/tennis-rl.git
cd tennis-rl
pip install pip install -r requirements.txt
```

3. Download the Unity Environment
Download the environment that matches your operation system, then place the file in the `tennis-rl/` folder and unizip the file.  

- [__Linux__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [__Mac OSX__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [__Windows (32-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [__Windows (64-bit)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)  
- [__GPU version (e.g. for AWS)__](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
Please note, that GPU version on AWS will only work if virtual screen is not enabled (you will not be able to watch the agent). 
If you would to do so, please follow the instructions in this [link](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)
and then download the environment for the Linux operating system above.  
	
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `name` environment.  
```bash
python -m ipykernel install --user --name name --display-name "name"
````

5. Before running code in a notebook, change the kernel to match the `name` environment by using the drop-down `Kernel` menu. 
  
## Setup of repository
Apart from the `Readme.md` and the `requirements.txt` file this repository consists of the following files:

1. `agent.py`: Agent and ReplayBuffer classes with all required functions
2. `model.py`: Actor and Critc Network classes
3. `run.py`: Script which will train the agent. Can be run directly from the terminal.
4. `report.ipynb`: As an alternative to the `run.py` script this Jupyter Notebook has a step-by-step structure. Here the learning algorithm is described in detail
5. `checkpoint_actor.pth`: Contains the weights of a successful Actor Network
6. `checkpoint_critic.pth`: Contains the weights of a successful Critic Network