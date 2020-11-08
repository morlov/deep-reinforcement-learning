# Project 1: Navigation

### Introduction

In this project we learn an agent to navigate (and collect bananas!) in a large, square world using Deep Q-Network algorithm.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

To install project environment



1. Create (and activate) a new environment with Python 3.6.
 
```bash
conda create --name drlnd python=3.6
source activate p1_navigation
```
    
2. Clone the repository and install dependencies.
```bash
git clone https://github.com/morlov/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install requirements.txt
```

3. Create an ipython kernel for the `p1_navigation` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "p1_navigation"
````

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

5. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions and content

Run the notebook Navigation.ipyng throughout to train and test agents. 

This notebook contains:
- DQNAgent - a class implementing Deep Q-network algorithm
- DQNPRAgent - a class implementing Deep Q-network with priority buffer algorithm
- A bunch of hepler function to run environment and check performance

You can aslo find saved network parameters for DQN agent in files:
- qnetwork_local.pt (local network weights)
- qnetwork_target.pt (target network weights)
      
As well as html version of notebok in:
- report.html (description of an algoritm used is provided in part 4)

### Futher work
Priority buffer algorithm is implemented in not optimal way as sampling with starightforward sampling. This can be improved by using sum-tree data structure to improve sampling time, that store experience samples in sorted by priority orde, which require O(log(N)) sampling complexity.

