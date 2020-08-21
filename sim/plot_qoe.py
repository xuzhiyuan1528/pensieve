import matplotlib.pyplot as plt
import numpy as np

f1dir = './results/log_test'
f2dir = './results-a3c/log_test'


def read_reward(fdir):
    reward = []
    with open(fdir) as fin:
        for row in fin:
            tmp = row.split('\t')
            print tmp
            reward.append(float(tmp[4]))
    return reward


reward1 = read_reward(f1dir)
reward2 = read_reward(f2dir)
idx = np.array(range(len(reward1))) * 100

plt.plot(idx[:25], reward1[:25], 'r', label='PnP-DRL')
plt.plot(idx[:25], reward2[:25], 'b', label='DRL-based')
plt.ylabel('Average QoE')
plt.xlabel('Time Step (Epoch)')
plt.legend()
plt.savefig('Two.eps')
plt.show()
