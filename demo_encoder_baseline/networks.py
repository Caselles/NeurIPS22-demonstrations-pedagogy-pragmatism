import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

'''
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

    ??????

'''


class DemoEncoderNetwork(nn.Module):
    def __init__(self, env_params, hidden_dim, goal_distrib_dim):
        super(DemoEncoderNetwork, self).__init__()

        self.max_length = env_params['max_demo_size']

        # Demo > predicted goal, predicted goal + agent trajectory > R

        # Demonstration encoder
        self.input_dim = env_params['obs'] +  env_params['action']
        self.hidden_dim = hidden_dim
        self.goal_distrib_dim = goal_distrib_dim
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.goal_distrib_dim)


    def encode(self, demos):

        lstm_out, _ = self.lstm(demos)
        x = F.relu(self.linear1(lstm_out[-1][-1].view(-1, self.hidden_dim)))
        output = self.output_layer(x)

        return output

    def forward(self, demos):

        lstm_out, _ = self.lstm(demos)
        x = F.relu(self.linear1(lstm_out[:,-1].view(-1, self.hidden_dim)))
        output = self.output_layer(x)
        

        #output = self.output_layer(lstm_out[:,-1].view(-1, self.hidden_dim))

        return output