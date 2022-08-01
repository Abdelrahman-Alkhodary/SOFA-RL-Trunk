import re
import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size-1)

		last_state = self.state[self.ptr-1]
		last_action = self.action[self.ptr-1]
		last_next_state = self.next_state[self.ptr-1]
		last_reward = self.reward[self.ptr-1]
		last_done = self.not_done[self.ptr-1] 
		
		return (
			torch.FloatTensor(np.vstack((self.state[ind], last_state))).to(self.device),
			torch.FloatTensor(np.vstack((self.action[ind], last_action))).to(self.device),
			torch.FloatTensor(np.vstack((self.next_state[ind], last_next_state))).to(self.device),
			torch.FloatTensor(np.vstack((self.reward[ind], last_reward))).to(self.device),
			torch.FloatTensor(np.vstack((self.not_done[ind], last_done))).to(self.device)
		)

'''
 The code of PER is modfied from https://github.com/BY571/TD3-and-Extensions/blob/main/scripts/buffer.py
'''

class PerReplayBuffer(object):
	def __init__(self, state_dim, action_dim, capacity=int(1e6), alpha=0.6, beta_start=0.4, beta_frames=100000):
		self.alpha = alpha
		self.beta_start = beta_start
		self.beta_frames = beta_frames
		self.frame = 1 # for beta calculation
		self.capacity = capacity
		self.size = 0
		self.ptr = 0
		self.priorities = np.zeros((capacity,), dtype=np.float32)

		self.state = np.zeros((self.capacity, state_dim))
		self.action = np.zeros((self.capacity, action_dim))
		self.next_state = np.zeros((self.capacity, state_dim))
		self.reward = np.zeros((self.capacity, 1))
		self.not_done = np.zeros((self.capacity, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def beta_by_frame(self, frame_idx):
		"""
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
		return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

	
	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		max_priority = self.priorities.max() if self.size > 0 else 1.0 # gives max priority if buffer is not empty else 1

		self.priorities[self.ptr] = max_priority

		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)


	def sample(self, batch_size):
		if self.size == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.ptr]

		# calcualte P = p^a/sum(p^a)
		probs = prios ** self.alpha
		P = probs / probs.sum()

		# get the indices of samples depending on probability P
		indices = np.random.choice(self.size, batch_size, p=P)

		beta = self.beta_by_frame(self.frame)
		self.frame += 1

		# compute importance sampling weight
		weights = (self.size * P[indices]) ** (-beta)
		# normalize weights
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)
		weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

		states = torch.FloatTensor(self.state[indices]).to(self.device)
		actions = torch.FloatTensor(self.action[indices]).to(self.device)
		next_states = torch.FloatTensor(self.next_state[indices]).to(self.device)
		rewards = torch.FloatTensor(self.reward[indices]).to(self.device)
		not_dones = torch.FloatTensor(self.not_done[indices]).to(self.device)

		return states, actions, next_states, rewards, not_dones, indices, weights


	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	
	def __len__(self):
		return self.size



