import copy 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 2048)
        self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.l2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(num_features=2048)
        self.l3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(num_features=2048)
        self.l4 = nn.Linear(2048, action_dim)

    def forward(self, state):
        a = F.relu(self.bn1(self.l1(state)))
        a = F.relu(self.bn2(self.l2(a)))
        a = F.relu(self.bn3(self.l3(a)))
        return torch.sigmoid(self.l4(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 1024)
        self.l5 = nn.Linear(1024, 1024)
        self.l6 = nn.Linear(1024, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005, policy_noise=0.2, noisy_clip=0.5, policy_freq=2):
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noisy_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        self.actor.eval()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        self.actor.train()

        #sample from replay buffer
        state, action, next_state, reward, not_done, indices, weights = replay_buffer.sample(batch_size)
         
        with torch.no_grad():
            # select action according to policy and add clipped noise
            ####################################
            noise = (torch.rand_like(action)*self.policy_noise).clamp(0, self.noise_clip)
            #################################
            next_action = (self.actor_target(next_state)+noise).clamp(0, 1)

            # compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # compute critic loss
        td1_error = current_Q1 - target_Q
        td2_error = current_Q2 - target_Q
        #critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_loss = 0.5* (td1_error.pow(2)*weights).mean() + 0.5* (td2_error.pow(2)*weights).mean()

        # update samples priorities
        errors = np.abs((torch.min(td1_error, td2_error)+1e-5).detach().cpu().numpy())
        replay_buffer.update_priorities(indices, errors)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        print('Networks weights are loaded')
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
