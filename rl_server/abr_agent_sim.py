import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6

class LSTM_ABR(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(LSTM_ABR, self).__init__()
        unit = 64

        self.q00 = nn.LSTM(1, unit, batch_first=True)
        self.q01 = nn.LSTM(1, unit, batch_first=True)
        self.q02 = nn.LSTM(1, unit, batch_first=True)
        self.q03 = nn.Linear(3, unit)

        self.q1 = nn.Linear(unit * 4, unit)
        self.q2 = nn.Linear(unit, num_actions)

        self.i00 = nn.LSTM(1, unit, batch_first=True)
        self.i01 = nn.LSTM(1, unit, batch_first=True)
        self.i02 = nn.LSTM(1, unit, batch_first=True)
        self.i03 = nn.Linear(3, unit)

        self.i1 = nn.Linear(unit * 4, unit)
        self.i2 = nn.Linear(unit, num_actions)

    def forward(self, state):
        (split_set1, split_set2) = state

        self.q00.flatten_parameters()
        _, (hn, _) = self.q00(split_set1[0])
        hn0 = torch.squeeze(hn, dim=0)

        self.q01.flatten_parameters()
        _, (hn, _) = self.q01(split_set1[1])
        hn1 = torch.squeeze(hn, dim=0)

        self.q01.flatten_parameters()
        _, (hn, _) = self.q01(split_set1[2])
        hn2 = torch.squeeze(hn, dim=0)

        hn3 = F.relu(self.q03(split_set2))

        hn = torch.cat([hn0, hn1, hn2, hn3], dim=1)
        q = F.relu(self.q1(hn))

        self.i00.flatten_parameters()
        _, (hn, _) = self.i00(split_set1[0])
        hn0 = torch.squeeze(hn, dim=0)

        self.i01.flatten_parameters()
        _, (hn, _) = self.i01(split_set1[1])
        hn1 = torch.squeeze(hn, dim=0)

        self.i02.flatten_parameters()
        _, (hn, _) = self.i02(split_set1[2])
        hn2 = torch.squeeze(hn, dim=0)

        hn3 = F.relu(self.i03(split_set2))

        hn = torch.cat([hn0, hn1, hn2, hn3], dim=1)
        i = F.relu(self.i1(hn))
        i = F.relu(self.i2(i))

        return self.q2(q), F.log_softmax(i, dim=1), i


class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        unit = 64
        self.q1 = nn.Linear(state_dim, unit)
        self.q2 = nn.Linear(unit, unit)
        self.q3 = nn.Linear(unit, num_actions)

        self.i1 = nn.Linear(state_dim, unit)
        self.i2 = nn.Linear(unit, unit)
        self.i3 = nn.Linear(unit, num_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))

        return self.q3(q), F.log_softmax(i, dim=1), i


def process_state2(states):
    return states


def process_state(states):
    if type(states) is not torch.Tensor:
        states = np.array(states)
        states = torch.from_numpy(states).float()

    states = states.reshape((-1, S_INFO, S_LEN))

    split_0 = states[:, 0:1, -1]
    split_1 = states[:, 1:2, -1]
    split_2 = states[:, 2:3, :].transpose(1, 2)
    split_3 = states[:, 3:4, :].transpose(1, 2)
    split_4 = states[:, 4:5, :A_DIM].transpose(1, 2)
    split_5 = states[:, 5:6, -1]

    set2 = torch.cat([split_0, split_1, split_5], dim=1)

    return (split_2, split_3, split_4), set2


class discrete_BCQ(object):
    def __init__(self):
        # fpath = '/home/eric/Data/drl-il/10015833-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> weighted*
        # fpath = '/home/eric/Data/drl-il/11171910-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> weighted
        # fpath = '/home/eric/Data/drl-il/13164312-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQ
        # fpath = '/home/eric/Data/drl-il/13164345-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQ
        # fpath = '/home/eric/Data/drl-il/13173559-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQ*
        # fpath = '/home/eric/Data/drl-il/16013337-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQx
        # fpath = '/home/eric/Data/drl-il/12012621-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQx
        # fpath = '/home/eric/Data/drl-il/12012234-sNone/mod/bcq_6673_Q' # 0.45 -> 0.0 -> BCQx

        # fpath = '/home/eric/Data/drl-il/10015128-sNone/mod/bcq_6673_Q' # 0.45 -> 0.01
        # fpath = '/home/eric/Data/drl-il/11180318-sNone/mod/bcq_6673_Q' # 0.45 -> 0.01

        # fpath = '/home/eric/Data/drl-il/10014644-sNone/mod/bcq_6673_Q' # 0.45 -> 0.03
        # fpath = '/home/eric/Data/drl-il/11235015-sNone/mod/bcq_6673_Q' # 0.45 -> 0.03

        # fpath = '/home/eric/Data/drl-il/10020743-sNone/mod/bcq_6673_Q' # 0.45 -> 0.05
        # fpath = '/home/eric/Data/drl-il/16012811-sNone/mod/bcq_6673_Q' # 0.45 -> 0.05 -> BCQ*
        # fpath = '/home/eric/Data/drl-il/16012113-sNone/mod/bcq_6673_Q' # 0.45 -> 0.05

        # fpath = '/home/eric/Data/drl-il/16013709-sNone/mod/bcq_6673_Q' # 0.45 -> 0.8 -> BCQ
        # fpath = '/home/eric/Data/drl-il/16014742-sNone/mod/bcq_6000_Q' # 0.45 -> 0.8


        # fpath = '/home/eric/Data/drl-il/10022511-sNone/mod/bcq_6673_Q' # 0.45 -> 0.1*
        # fpath = '/home/eric/Data/drl-il/15212244-sNone/mod/bcq_6673_Q' # 0.45 -> 0.1*
        # fpath = '/home/eric/Data/drl-il/16010254-sNone/mod/bcq_6673_Q' # 0.45 -> 0.1 -> BCQ*
        # fpath = '/home/eric/Data/drl-il/16005256-sNone/mod/bcq_2000_Q' # 0.45 -> 0.1

        # fpath = '/home/eric/Data/drl-il/13170913-sNone/mod/bcq_6673_Q' # 0.45 -> 0.5 -> BCQ*
        # fpath = '/home/eric/Data/drl-il/12013649-sNone/mod/bcq_6673_Q' # 0.45 -> 0.5 - > BCQx
        # fpath = '/home/eric/Data/drl-il/12021052-sNone/mod/bcq_6673_Q' # 0.45 -> 0.5 - > BCQx
        # fpath = '/home/eric/Data/drl-il/10023455-sNone/mod/bcq_2000_Q' # 0.45 -> 0.5
        # fpath = '/home/eric/Data/drl-il/12010402-sNone/mod/bcq_6673_Q' # 0.45 -> 0.5 *

        # fpath = '/home/eric/Data/drl-il/10165519-sNone/mod/bcq_10000_Q' # 0.45 -> rl -> weights - 3G
        # fpath = '/home/eric/Data/drl-il/11044737-sNone/mod/bcq_11000_Q' # 0.45 -> rl -> weights - 3G

        fpath = '/home/eric/Data/drl-il/16020447-sNone/mod/bcq_13347_Q' # from Baselines
        # fpath = '/home/eric/Data/drl-il/16050054-sNone/mod/bcq_13347_Q' # from Baselines
        # fpath = '/home/eric/Data/drl-il/16043623-sNone/mod/bcq_13347_Q' # from Baselines
        # fpath = '/home/eric/Data/drl-il/16031626-sNone/mod/bcq_13347_Q' # from Baselines BCQ

        self.threshold = 0.45

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = S_INFO * S_LEN
        num_actions = A_DIM

        self.Q = LSTM_ABR(state_dim, num_actions).to(self.device)
        # self.Q = FC_Q(state_dim, num_actions).to(self.device)

        self.state_shape = (-1, state_dim)
        print(os.getcwd())
        self.Q.load_state_dict(torch.load(fpath, map_location=self.device))
        print('load mode from', fpath)

        self.Q.eval()

    def select_action(self, state):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action

        with torch.no_grad():
            state = torch.from_numpy(state).float().reshape(self.state_shape).to(self.device)
            state = process_state(state)
            q, imt, i = self.Q(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax
            return int((imt * q + (1. - imt) * -1e8).argmax(1))


if __name__ == '__main__':
    import numpy as np
    model = discrete_BCQ()
    state = np.random.random(48)
    action = model.select_action(state)
    print('select action', action)
