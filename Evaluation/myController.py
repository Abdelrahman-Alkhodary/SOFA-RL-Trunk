from math import sqrt
from os import curdir
import random
import Sofa
import numpy as np
import pandas as pd
import torch
from TD3 import TD3
#from utils import HerReplayBuffer
import logging

""" The state will be the positions of the goal and the end effector along with the eight cables actuation"""

''' The reward function will be the increase or decrease of the distance between goal and end effector 
plus negative the total steps taken from the start of the episode plus huge reward when the distance is within small range'''

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


# This python script shows the functions to be implemented
# in order to create your Controller in python
class TrunkController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        # These are needed (and the normal way to override from a python class)
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        #############################################################
        ##################### SOFA ##################################
        #############################################################

        self.node = kwargs["node"]
        
        # Control the on begin animation to enter every 10 animation step <<<<<<<<<
        self.iteration = 0
        self.maxIeration = 500
        
        # Accessing the child of the rootNode
        self.Simulation=self.node.getChild("Simulation")
        self.Trunk=self.Simulation.getChild("Trunk")
        self.Effector = self.Trunk.getChild('Effectors')
        self.Goal = self.node.getChild("Goal")
            
        
        # Get the cables to the end of the parent Trunk
        self.cableL0=self.Trunk.getChild('cableL0')
        self.cableL1=self.Trunk.getChild('cableL1')
        self.cableL2=self.Trunk.getChild('cableL2')
        self.cableL3=self.Trunk.getChild('cableL3')
        
        self.cableS0=self.Trunk.getChild('cableS0')
        self.cableS1=self.Trunk.getChild('cableS1')
        self.cableS2=self.Trunk.getChild('cableS2')
        self.cableS3=self.Trunk.getChild('cableS3')
        
        # you can access the object through its name
        self.cableL0Acutator = self.cableL0.cable
        self.cableL1Acutator = self.cableL1.cable
        self.cableL2Acutator = self.cableL2.cable
        self.cableL3Acutator = self.cableL3.cable
        
        self.cableS0Acutator = self.cableS0.cable
        self.cableS1Acutator = self.cableS1.cable
        self.cableS2Acutator = self.cableS2.cable
        self.cableS3Acutator = self.cableS3.cable

        logging.basicConfig(filename="deubg.log", format='%(asctime)s %(message)s', filemode='w')
        #Let us Create an object 
        self.logger=logging.getLogger() 
        #Now we are going to Set the threshold of logger to DEBUG 
        self.logger.setLevel(logging.DEBUG)
        
        
        # To get the position of the end effector
        self.endEffector = self.Effector.myEndEffector
        self.effectorPosition = self.endEffector.findData('position').value
        
        # To get the position of the goal
        self.goal_mech = self.Goal.goalPos
        self.goalPosition = self.goal_mech.findData('position').value
        self.logger.debug(f'The goals position x, y, z: {self.goalPosition}')

        #############################################################
        ##################### RL ####################################
        #############################################################
        
        # The RL TD3 policy
        self.state_dim = 14
        self.action_dim = 8
        self.policy = TD3(self.state_dim, self.action_dim)

        # load the previous trained models
        self.policy.load('./weights/weights')

        # noise to add to the action for exploration
        self.batch_size = 256
        # this will track the rewards inside one episode
        self.episode_rewards = []
        # this will have the reward of each step in all episodes
        self.all_episodes_rewards = []
        # this will have total epsiodes rewards
        self.rewards = []
        # flag to begin in the starting state
        self.starting_state = True

        # normalization
        # length of the arm is 195 in direction of z and 0 in direction of x & y 
        # the arm covers a half sphere so theoritically the max distance between end eff and goal will be the diameter of the sphere 
        # the min distance will be 0 
        self.max_distance = 195

        # counter of total episodes
        self.episode_itr = 0
        self.save_weights_episode = 500
        # counter for total number of steps inside one episode
        self.step_itr = 0
        self.max_step_itr = 200
        # counter for the reset period to differntiate between two consecutive episodes
        self.reset_itr = 0
        # counter for total number of steps taken from the start
        self.steps_total = 0

        
        self.episode_reward = 0
        self.episode_done = True

        # times to reach the goal
        self.goal_reached_count = 0
        # number of set goals
        self.number_of_goals = 0
        # trainig status
        self.training_status = {}

                     


    def onAnimateBeginEvent(self, eventType):
        if self.starting_state:
            # The state will be the end effector and goal positions along with the eight cable actuations
            self.state = self.get_state()
            self.logger.debug(f'The state: {self.state}')
            self.action = self.policy.select_action(self.state)
            self.logger.debug(f'The action: {self.action}')
            
            zero_one_output = self.action # scales (0,1)
            # now scale to the environment's action space 
            # action = zero_one_output * max value of each action
            action = [a * b for a, b in zip([70.0, 70.0, 70.0, 70.0, 40.0, 40.0, 40.0, 40.0], zero_one_output)]

            
            self.cableL0Acutator.findData('displacement').value = action[0]
            self.cableL1Acutator.findData('displacement').value = action[1] 
            self.cableL2Acutator.findData('displacement').value = action[2]
            self.cableL3Acutator.findData('displacement').value = action[3]
            
            self.cableS0Acutator.findData('displacement').value = action[4]
            self.cableS1Acutator.findData('displacement').value = action[5]
            self.cableS2Acutator.findData('displacement').value = action[6]
            self.cableS3Acutator.findData('displacement').value = action[7]

            self.starting_state = False
            self.episode_done = False

        elif not self.episode_done:
            self.step_itr += 1
            self.next_state = self.get_state()

            if self.distance() < 5:
                self.goal_reached_count += 1
                print(f'Number {self.goal_reached_count} goal is reached at the step: {self.step_itr} of the episode: {self.episode_itr}')
                self.episode_done = True
                self.step_itr = 0

            self.state = self.next_state
            self.action = self.policy.select_action(self.state)
            self.logger.debug(f'The action: {self.action}')
            
            # rescaling actions to normal range
            zero_one_output = self.action # scales (0,1)
            # now scale to the environment's action space 
            # action = zero_one_output * max value of each action 
            action = [a * b for a, b in zip([70.0, 70.0, 70.0, 70.0, 40.0, 40.0, 40.0, 40.0], zero_one_output)]
            
            self.cableL0Acutator.findData('displacement').value = action[0]
            self.cableL1Acutator.findData('displacement').value = action[1] 
            self.cableL2Acutator.findData('displacement').value = action[2]
            self.cableL3Acutator.findData('displacement').value = action[3]
            
            self.cableS0Acutator.findData('displacement').value = action[4]
            self.cableS1Acutator.findData('displacement').value = action[5]
            self.cableS2Acutator.findData('displacement').value = action[6]
            self.cableS3Acutator.findData('displacement').value = action[7]

            if self.step_itr >= self.max_step_itr:
                print(f'Episode {self.episode_itr} is finished')
                self.step_itr = 0
                self.episode_done = True
        else:
            Sofa.Simulation.reset(self.node)
            self.reset()
            goal_positions_df = pd.read_csv('evaluating_goals.csv')
            _, x, y, z = goal_positions_df.iloc[random.randrange(0,len(goal_positions_df)),:]
            self.logger.debug(f'The goals position x, y, z: {[[x, y, z]]}')
            self.goal_mech.findData('position').value = [[x, y, z]] 
            self.episode_done = False
            self.starting_state = True
            self.step_itr = 0
            self.episode_itr += 1

            
    	        
    	
    def onKeypressedEvent(self, event):
        key = event['key']
        if key == '1':  
            self.cableL0Acutator.findData('displacement').value += 2.0
            print(self.cableL0Acutator.findData('displacement').value)
            effectorPosition = self.endEffector.findData('position').value
            print(f'effectorPosition = {effectorPosition}')

        if ord(key) == 19:  # up
            print("***********************************")
            #effectorPosition = self.endEffector.findData('position').value
            #print(f'effectorPosition = {effectorPosition}')
            print(f'** Number of reached goals {self.goal_reached_count} **')
            print("***********************************")
            

            
    
    def reset(self):
        # reset the displacement in the cables
        self.cableL0Acutator.findData('displacement').value = 0
        self.cableL1Acutator.findData('displacement').value = 0
        self.cableL2Acutator.findData('displacement').value = 0
        self.cableL3Acutator.findData('displacement').value = 0

        self.cableS0Acutator.findData('displacement').value = 0
        self.cableS1Acutator.findData('displacement').value = 0
        self.cableS2Acutator.findData('displacement').value = 0
        self.cableS3Acutator.findData('displacement').value = 0

        # reset the length of the cables
        self.cableL0Acutator.findData('cableLength').value = 185
        self.cableL1Acutator.findData('cableLength').value = 185
        self.cableL2Acutator.findData('cableLength').value = 185
        self.cableL3Acutator.findData('cableLength').value = 185

        self.cableS0Acutator.findData('cableLength').value = 91
        self.cableS1Acutator.findData('cableLength').value = 91
        self.cableS2Acutator.findData('cableLength').value = 91
        self.cableS3Acutator.findData('cableLength').value = 91



    def get_state(self):
        # The state will be the end effector and goal positions along with the eight cable actuations
        # preprocessing the input by subtracting the minimum and dividing by the range 
        # the minimum is zero in distance and in cable actuation so the normalization will only be dividing by the max
        # extracting the end effector and goal positions and dividing by 195 to normalize 
        endEffectorPos = self.endEffector.findData('position').value
        endEffectorPos_X, endEffectorPos_Y, endEffectorPos_Z = endEffectorPos[0][0]/195, endEffectorPos[0][1]/195, endEffectorPos[0][2]/195
        goalPos = self.goal_mech.findData('position').value
        #print(f'Inside get_state goalPos: {goalPos}')
        goalPos_X, goalPos_Y, goalPos_Z = goalPos[0][0]/195, goalPos[0][1]/195, goalPos[0][2]/195
        # the actuations' values
        cableDispL0 = self.cableL0Acutator.findData('displacement').value
        cableDispL1 = self.cableL1Acutator.findData('displacement').value
        cableDispL2 = self.cableL2Acutator.findData('displacement').value
        cableDispL3 = self.cableL3Acutator.findData('displacement').value
        
        cableDispS0 = self.cableS0Acutator.findData('displacement').value
        cableDispS1 = self.cableS1Acutator.findData('displacement').value
        cableDispS2 = self.cableS2Acutator.findData('displacement').value
        cableDispS3 = self.cableS3Acutator.findData('displacement').value
        
        cableDispL0_norm = cableDispL0 / 70
        cableDispL1_norm = cableDispL1 / 70
        cableDispL2_norm = cableDispL2 / 70
        cableDispL3_norm = cableDispL3 / 70
        
        cableDispS0_norm = cableDispS0 / 40
        cableDispS1_norm = cableDispS1 / 40
        cableDispS2_norm = cableDispS2 / 40
        cableDispS3_norm = cableDispS3 / 40

        return np.array([endEffectorPos_X, endEffectorPos_Y, endEffectorPos_Z, goalPos_X, goalPos_Y, goalPos_Z, cableDispL0_norm, cableDispL1_norm, cableDispL2_norm, cableDispL3_norm, cableDispS0_norm, cableDispS1_norm, cableDispS2_norm, cableDispS3_norm])

    def reward_func(self, steps):
        # define the weights of reward function components
        #alpha = 0.01     # weight of distance
        #beta = 0.1        # weight of number of steps taken in one episode
        #gamma = 0.5       # weight of number of active cables
        #omega = 1       # weight of reward for close distance to the goal
        '''
        cableDispL0 = self.cableL0Acutator.findData('displacement').value
        cableDispL1 = self.cableL1Acutator.findData('displacement').value
        cableDispL2 = self.cableL2Acutator.findData('displacement').value
        cableDispL3 = self.cableL3Acutator.findData('displacement').value
        cableDispS0 = self.cableS0Acutator.findData('displacement').value
        cableDispS1 = self.cableS1Acutator.findData('displacement').value
        cableDispS2 = self.cableS2Acutator.findData('displacement').value
        cableDispS3 = self.cableS3Acutator.findData('displacement').value

        cables = [cableDispL0, cableDispL1, cableDispL2, cableDispL3, cableDispS0, cableDispS1, cableDispS2, cableDispS3]

        active_cables = 0
        for cable in cables:
            if cable > 1:
                active_cables += 1
        '''
        current_distance = self.distance()
        #print(f'neg the distance: {- (current_distance / self.max_distance)}')
        return  - current_distance 

    def distance(self):
        self.goalPosition = self.goal_mech.findData('position').value
        eff_goal_dist = sqrt((self.goalPosition[0][0] - self.effectorPosition[0][0]) ** 2 +
        (self.goalPosition[0][1] - self.effectorPosition[0][1]) ** 2 +
        (self.goalPosition[0][2] - self.effectorPosition[0][2]) ** 2) 
        return eff_goal_dist
