# P-DQN-Proposed_state_reward
This repo consists all the files related to PDQN proposed reward and state
# Theory
This is a traffic signal controller based on P-DQN.
P-DQN-Paramterized Deep Q Network - Family of model free, hybrid( Has Q network for discret action - value based and Parameter network for continuous action - policy based) deep reinforcement learning algorithms.
# Environment
A 2X2 grid with 4 intersections and 32 lanes. Each edge has 2 incoming lanes and 2 outgoing lanes. The leftmost incoming lane is to turn left only. The right most lane is to turn right and go straight as well.
# State Definition
Each lane is divided into a vector of 10 cells. Each cell indicates a floating point value indicating two things:
1. The presence of vehicle.
2. The normlaized speed of the vechicle
For example, if the vehicle is absent, the cell value would be 0 and if there is a vehicle and it is stationary, the value would be 1 else, value would be greater than 1 indicating the motion of the vehicle.
# Reward Definition
We have used both waiting times and queue lengths in framing the reward.
reward = (old_total_wait) * (old_queue/max_queue) - (current_total_wait) * (current_queue/max_queue)
Here old_total_wait and current_total_wait refer to waiting times in previous and current simulations. Similarly, old_queue and current_queue refer to queue lengths in previous and current simulations.
# Action Space - discrete action - selected by Q network
It is dicrete and is comprised of 4 signal phases
1. NSG - North South Green - Green signal to vehicles travelling in North and South lanes and travel straight or right.
2. NSLG - North South Left Green - Green signal to vehicles travelling in North and South lanes and travel left.
3. EWG - East West Green - Green signal to vehicles travelling in East and West lanes and travel straight or right.
4. EWLG - East West Left Green - Green signal to vehicles travelling in East and West lanes and travel left.
# Action space - continuous action - selcted by parameter network
Duration of the the phase selected by Q network.
# Requirments to run the code
1. Install SUMO simulation tool.
2. Install anaconda.
3. After installing anaconda, open anaconda prompt. There, create a virtual environment from where you will run the codes as follows:
conda create --name tf_gpu
activate tf_gpu
conda install tensorflow-gpu
4. Use this tf_gpu environment to run your code, here run pip install requirements.txt to install all the required packages.
5. Just run the training_main.py code to see the code working.
6. To see SUMO GUI, set gui to True in training_settings.ini file.
7. Note, if you want to change the network, you have to change the contents of the training_simulation.py, training_settings.ini, utils.py, testing_simulation.py, testing_settings.ini files as well.
8. Once you run the code, models folder will be created where you can see the stats of each trained module.
