import matplotlib.pyplot as plt
import os
import glob
import numpy as np

class AlgResults():
    def __init__(self, alg_name):
        self.alg_name = alg_name
        self.fnames = []
        self.states = []
        self.rewards = []
        self.ave_rewards_each_trace = []
        self.ave_rewards_all_trace = 0.0
        self.stds_rewards_all_trace = 0.0
        self.actions = []
        self.nxstates = []
        self.notdones = []

    def cal_rewards(self, norm_base=1.0):
        self.ave_rewards_each_trace = []
        self.ave_rewards_all_trace = 0.0
        self.stds_rewards_all_trace = 0.0
        total_trace = 0

        for trace_reward, fname in zip(self.rewards, self.fnames):
            total_trace_reward = 0.0
            total_line = 0

            for line_reward in trace_reward:
                total_trace_reward += line_reward
                total_line += 1

            if total_line == 0:
                print('%s: %s file has no data!' % (self.alg_name, fname))
                continue

            total_trace += 1
            ave_ward_each_trace = total_trace_reward / float(total_line)

            ave_ward_each_trace /= norm_base

            self.ave_rewards_each_trace.append(ave_ward_each_trace)
            self.ave_rewards_all_trace += ave_ward_each_trace

        if total_trace == 0:
            print('%s has no trace!' % (self.alg_name))
        else:
            self.ave_rewards_all_trace /= float(total_trace)
            self.ave_rewards_each_trace = np.array(self.ave_rewards_each_trace)
            self.stds_rewards_all_trace = np.std(self.ave_rewards_each_trace)
            # all_rewards = np.array([])
            # for cur_reward in self.rewards:
            #     all_rewards = np.append(all_rewards, np.array(cur_reward))
            # self.stds_rewards_all_trace = np.std(all_rewards)

def deal_with_pensieve(fname):
    state = []
    action = []
    nxstate = []
    reward = []
    notdone = []
    with open(os.path.join(base_dir, fname), 'r') as fin:
        for row in fin:
            tmp = row.split('|')
            if len(tmp) < 5 or float(tmp[3]) == 0.0:
                continue

            sep = ',' if ',' in tmp[0] else ' '
            z = tmp[0][1:-2]
            state.append(np.fromstring(tmp[0][1:-1], dtype=np.float, sep=sep))

            sep = ',' if ',' in tmp[1] else ' '
            action.append(np.fromstring(tmp[1][1:-1], dtype=np.float, sep=sep))
            action[-1] = np.argmax(action[-1])

            sep = ',' if ',' in tmp[2] else ' '
            nxstate.append(np.fromstring(tmp[2].replace('[', '').replace(']', '').strip(), dtype=np.float, sep=sep))
            reward.append(float(tmp[3]))
            notdone.append(1 - (0 if 'False' in tmp[4] else 1))

    return state, action, reward, nxstate, notdone

# base_dir = '/home/eric/Dropbox/Projects-Research/0-DRL-Imitation/Pensieve_emu_304mbps'
# base_dir = '/home/eric/Dev/DRL-IL/pensieve/run_exp/results'
# base_dir = '/home/cst/wk/Pensieve/data/norway_results/train_results/results'
# base_dir = '/home/cst/wk/Pensieve/data/norway_results/test_results/results'
base_dir = '/home/eric/Dropbox/Projects-Research/0-DRL-Imitation/norway_results/train_results/results'
# base_dir = '/home/cst/wk/Pensieve/data/results_7772mbps_20200805/results'

# alg_names = ['BOLA', 'fastMPC', 'robustMPC', 'Our', 'RL']
alg_names = ['BOLA', 'fastMPC', 'robustMPC', 'RL', 'BB', 'Ours']
# alg_names = ['Ours']
alg_results = dict()

for alg_name in alg_names:
    alg_results[alg_name] = AlgResults(alg_name)

for fname in sorted(os.listdir(base_dir)):
    # if 'trace_0' not in fname:
    #     continue
    # print(f'process file {fname}')
    if 'npy' in fname:
        continue
    alg_name = fname.split('_')[1]

    cur_alg_results = alg_results[alg_name]

    state, action, reward, nxstate, notdone = deal_with_pensieve(fname)

    cur_alg_results.fnames.append(fname)
    cur_alg_results.states.append(state)
    cur_alg_results.actions.append(action)
    cur_alg_results.rewards.append(reward)
    cur_alg_results.nxstates.append(nxstate)
    cur_alg_results.notdones.append(notdone)

alg_results['RL'].cal_rewards()
norm_base =alg_results['RL'].ave_rewards_all_trace
for alg_name in alg_names:
    # if alg_name == 'RL':
    #     continue
    alg_results[alg_name].cal_rewards(norm_base=norm_base)

############## plot figure #########

SMALL_SIZE = 8
MEDIUM_SIZE = 11
LARGE_SIZE = 14
LARGER_SIZE = 16

plt.rc('font', size=LARGE_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LARGE_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

# plot Inference Latency for fps 53
index = np.arange(len(alg_names))
# names = ['CPU', 'GPU', 'Round-Robin', 'COSREL-P', 'COSREL-E']
names = alg_names
labels = alg_names
color = ['#8437A6', '#4285F4', '#FBBC05', '#EA4335', '#34A853']
error_kw = {'ecolor':'k', 'capsize':3, 'elinewidth':2}
patterns = ('|', '-', 'x', '//', '\\\\')
show = True

plt.figure()
plt.grid(linestyle='--', linewidth=0.5, axis='y', dashes=(10, 5))
ax = plt.gca()
ax.set_axisbelow(True)
# avgs = [avg_times_list_cpu[0], avg_times_list_gpu[0], avg_times_list_rr[0], avg_times_list_dqn[0], avg_times_list_dqnp[0]]
# stds = [std_times_list_cpu[0], std_times_list_gpu[0], std_times_list_rr[0], std_times_list_dqn[0], std_times_list_dqnp[0]]
# avgs = np.random.rand(5)
# stds = np.random.rand(5)
avgs = np.zeros(len(alg_names), dtype=float)
stds = np.zeros(len(alg_names), dtype=float)
for i, alg_name in enumerate(alg_names):
    avgs[i] = alg_results[alg_name].ave_rewards_all_trace
    stds[i] = alg_results[alg_name].stds_rewards_all_trace

print('Latency')
print(avgs)
print(stds)

# plt.title('Inference Latency')
plt.ylabel('Normalized average QoE')
# plt.bar(index, avgs, yerr=stds, color=color, error_kw=error_kw, label=labels, hatch=patterns)
hatch_density = 2
for i, pattern in enumerate(patterns[:len(alg_names)]):
    plt.bar(i, avgs[i], hatch=pattern * hatch_density,
            label=labels[i], color=color[i], yerr=stds[i], error_kw=error_kw, width=1.0)

plt.xticks(index, names)
# plt.legend()
# plt.legend(bbox_to_anchor=(0.9, 1.10), ncol=len(alg_names), fontsize=12)
plt.legend(loc='lower left', bbox_to_anchor=(0., 0.98, 1., .08), mode='expand', ncol=len(alg_names), fontsize=13, frameon=False)
if show:
    plt.show()
    plt.close()
else:
    # plt.savefig('Normalized average QoE.eps', format='eps')
    plt.savefig('Normalized average QoE.jpg')