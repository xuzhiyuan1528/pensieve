import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, 128)
        self.q1_bn = nn.BatchNorm1d(128)
        self.q2 = nn.Linear(128, 128)
        self.q2_bn = nn.BatchNorm1d(128)
        # self.q2_dp = nn.Dropout(0.3)
        self.q3 = nn.Linear(128, num_actions)

        self.i1 = nn.Linear(state_dim, 128)
        self.i1_bn = nn.BatchNorm1d(128)
        self.i2 = nn.Linear(128, 128)
        self.i2_bn = nn.BatchNorm1d(128)
        self.i3 = nn.Linear(128, num_actions)
        self.i3_bn = nn.BatchNorm1d(num_actions)

    def forward(self, state):
        q = F.relu(self.q1_bn(self.q1(state)))
        q = F.relu(self.q2_bn(self.q2(q)))

        i = F.relu(self.i1_bn(self.i1(state)))
        i = F.relu(self.i2_bn(self.i2(i)))
        i = F.relu(self.i3_bn(self.i3(i)))
        return self.q3(q), F.log_softmax(i, dim=1), i


class discrete_BCQ(object):
    def __init__(self, fpath):
        self.device = 'cpu'
        state_dim = 48
        num_actions = 6

        self.Q = FC_Q(state_dim, num_actions).to(self.device)

        self.state_shape = (-1, state_dim)
        self.threshold = 0.3
        print(os.getcwd())
        self.Q.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))
        print('load mode from', fpath)

        self.Q.eval()

    def select_action(self, state):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        with torch.no_grad():
            state = torch.from_numpy(state).float().reshape(self.state_shape).to(self.device)
            q, imt, i = self.Q(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax
            return int((imt * q + (1. - imt) * -1e8).argmax(1))


if __name__ == '__main__':
    import numpy as np
    model = discrete_BCQ('./bcq_model.pt')
    state = np.random.random(48)
    action = model.select_action(state)
    print('select action', action)
