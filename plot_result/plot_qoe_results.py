import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# base_dir = '/home/eric/Dropbox/Projects-Research/0-DRL-Imitation/Pensieve_emu_304mbps'
# base_dir = '/home/eric/Dev/DRL-IL/pensieve/run_exp/results'
# base_dir = '/home/cst/wk/Pensieve/data/norway_results/train_results/results'
# base_dir = '/home/cst/wk/Pensieve/data/results_7772mbps_20200805/results'

# root = '/home/cst/Dropbox/0-DRL-Imitation/norway_results/train_results'
# root = '/home/cst/Dropbox/0-DRL-Imitation/norway_results/test_results'
root = '/home/cst/Dropbox/0-DRL-Imitation/link304-results'
# root = '/home/cst/Dropbox/0-DRL-Imitation/sim_results'
base_dir = os.path.join(root, 'results')
ori_base_dir = os.path.join(root, 'results_ori')

SIM_FLAG = False
if root.find('sim_') >= 0:
    SIM_FLAG = True
    base_dir = os.path.join(root, 'gen-traces')
    ori_base_dir = os.path.join(root, 'gen-logs')

VIDEO_LEN = 48
USE_Sigmoild = False

class AlgResults():
    def __init__(self, alg_name):
        self.alg_name = alg_name
        self.fnames = []
        self.states = []
        self.rewards = []
        self.total_rewards_each_trace = []
        self.ave_rewards_all_trace = 0.0
        self.stds_rewards_all_trace = 0.0
        self.actions = []
        self.nxstates = []
        self.notdones = []

        self.ori_fnames = []
        self.ori_rewards = []
        self.ori_total_rewards_each_trace = []
        self.ori_ave_rewards_all_trace = 0.0
        self.ori_stds_rewards_all_trace = 0.0

    def cal_rewards(self, norm_base=1.0):
        self.total_rewards_each_trace = []
        self.ave_rewards_all_trace = 0.0
        self.stds_rewards_all_trace = 0.0
        total_trace = 0

        for trace_reward, fname in zip(self.rewards, self.fnames):
            # total_trace_reward = 0.0
            # total_line = 0
            #
            # for line_reward in trace_reward:
            #     total_trace_reward += line_reward
            #     total_line += 1
            if USE_Sigmoild:
                trace_reward = 1 / (1 + np.exp(-np.array(trace_reward)))

            total_trace_reward = np.sum(trace_reward) / norm_base
            # total_trace_reward = np.mean(trace_reward) / norm_base
            total_line = len(trace_reward)

            if total_line == 0:
                print('%s: %s file has no data!' % (self.alg_name, fname))
                continue

            total_trace += 1
            # ave_ward_each_trace = total_trace_reward / float(total_line)
            # ave_ward_each_trace /= norm_base

            self.total_rewards_each_trace.append(total_trace_reward)
            # self.ave_rewards_all_trace += total_trace_reward

        if total_trace == 0:
            print('%s has no trace!' % (self.alg_name))
        else:
            self.total_rewards_each_trace = np.array(self.total_rewards_each_trace)
            self.ave_rewards_all_trace = np.mean(self.total_rewards_each_trace)
            self.stds_rewards_all_trace = np.std(self.total_rewards_each_trace)
            # all_rewards = np.array([])
            # for cur_reward in self.rewards:
            #     all_rewards = np.append(all_rewards, np.array(cur_reward))
            # self.stds_rewards_all_trace = np.std(all_rewards)

    def cal_ori_rewards(self, norm_base=1.0):
        self.ori_total_rewards_each_trace = []
        self.ori_ave_rewards_all_trace = 0.0
        self.ori_stds_rewards_all_trace = 0.0
        total_trace = 0

        for trace_reward, fname in zip(self.ori_rewards, self.ori_fnames):
            # total_trace_reward = 0.0
            # total_line = 0

            if USE_Sigmoild:
                trace_reward = 1 / (1 + np.exp(-np.array(trace_reward)))

            total_trace_reward = np.sum(trace_reward) / norm_base
            total_line = len(trace_reward)

            # for line_reward in trace_reward:
            #     total_trace_reward += line_reward
            #     total_line += 1

            if total_line == 0:
                print('%s: %s file has no data!' % (self.alg_name, fname))
                continue

            total_trace += 1
            # ave_ward_each_trace = total_trace_reward / float(total_line)
            # ave_ward_each_trace /= norm_base

            self.ori_total_rewards_each_trace.append(total_trace_reward)
            # self.ave_rewards_all_trace += total_trace_reward

        if total_trace == 0:
            print('%s has no trace!' % (self.alg_name))
        else:
            self.ori_total_rewards_each_trace = np.array(self.ori_total_rewards_each_trace)
            self.ori_ave_rewards_all_trace = np.mean(self.ori_total_rewards_each_trace)
            self.ori_stds_rewards_all_trace = np.std(self.ori_total_rewards_each_trace)
            # all_rewards = np.array([])
            # for cur_reward in self.rewards:
            #     all_rewards = np.append(all_rewards, np.array(cur_reward))
            # self.stds_rewards_all_trace = np.std(all_rewards)

    def deal_with_pensieve(self, fname, old_reward_flag):
        state = []
        action = []
        nxstate = []
        reward = []
        notdone = []
        with open(os.path.join(base_dir, fname), 'r') as fin:
            for row in fin:
                tmp = row.split('|')
                # if len(tmp) < 5 or float(tmp[3]) == 0.0:
                #     continue

                if len(tmp) < 5:
                    continue

                sep = ',' if ',' in tmp[0] else ' '
                state.append(np.fromstring(tmp[0][1:-1], dtype=np.float, sep=sep))

                sep = ',' if ',' in tmp[1] else ' '
                action.append(np.fromstring(tmp[1][1:-1], dtype=np.float, sep=sep))
                action[-1] = np.argmax(action[-1])

                sep = ',' if ',' in tmp[2] else ' '
                nxstate.append(np.fromstring(tmp[2].replace('[', '').replace(']', '').strip(), dtype=np.float, sep=sep))
                reward.append(float(tmp[3]))
                notdone.append(1 - (0 if 'False' in tmp[4] else 1))

        if old_reward_flag:
            start = 2
            end = min(len(reward), VIDEO_LEN)
        else:
            start = 1
            end = min(len(reward), VIDEO_LEN)

        # if len(reward) < VIDEO_LEN:
        #     return

        self.fnames.append(fname)
        self.states.append(state[start:end])
        self.actions.append(action[start:end])
        self.rewards.append(reward[start:end])
        self.nxstates.append(nxstate[start:end])
        self.notdones.append(notdone[start:end])
        # return state, action, reward, nxstate, notdone

    def deal_with_ori_pensieve(self, ori_fname, old_reward_flag):
        reward = []
        with open(os.path.join(ori_base_dir, ori_fname), 'r') as fin:
            for line in fin:
                parse = line.split()
                if len(parse) <= 1:
                    continue
                # time_ms.append(float(parse[0]))
                # bit_rate.append(int(parse[1]))
                # buff.append(float(parse[2]))
                # bw.append(float(parse[4]) / float(parse[5]) *
                #           BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                reward.append(float(parse[6]))

        if old_reward_flag:
            start = 1
            end = min(len(reward)-1, VIDEO_LEN-1)
        else:
            start = 1
            end = min(len(reward), VIDEO_LEN)

        # if len(reward) < VIDEO_LEN:
        #     return

        self.ori_fnames.append(fname)
        self.ori_rewards.append(reward[start:end])

# root = '/home/cst/wk/Pensieve/data/norway_results/train_results/'
# base_dir = os.path.join(root, 'results')
#
# our_result_dir = os.path.join(root, 'RL-best')
# our_state_path = os.path.join(our_result_dir, 'sample-state.npy')
# our_action_path = os.path.join(our_result_dir, 'sample-action.npy')
# our_reward_path = os.path.join(our_result_dir, 'sample-reward.npy')
#
# our_state = np.load(our_state_path)
# our_action = np.load(our_action_path)
# our_reward = np.load(our_reward_path)

# alg_names = ['BOLA', 'fastMPC', 'robustMPC', 'Our', 'RL']
alg_names = ['BB', 'fastMPC', 'robustMPC', 'BOLA', 'Ours', 'RL']
if root.find('link304') >= 0:
    alg_names = ['fastMPC', 'robustMPC', 'BOLA', 'Ours', 'RL']
if SIM_FLAG:
    alg_names = ['fastMPC', 'robustMPC', 'BOLA', 'Ours', 'RL']

alg_results = dict()

for alg_name in alg_names:
    alg_results[alg_name] = AlgResults(alg_name)

for fname in sorted(os.listdir(base_dir)):
    # if 'trace_0' not in fname:
    #     continue
    # print(f'process file {fname}')
    if fname.find('.npy') >= 0:
        continue
    alg_name = fname.split('_')[1]

    cur_alg_results = alg_results[alg_name]

    if alg_name in ['BOLA', 'BB']:
        old_reward_flag = True
    else:
        old_reward_flag = False

    cur_alg_results.deal_with_pensieve(fname, old_reward_flag)
    # cur_alg_results.deal_with_ori_pensieve(fname, old_reward_flag)

alg_results['RL'].cal_rewards()
norm_base = alg_results['RL'].ave_rewards_all_trace
# norm_base = 1.0
for alg_name in alg_names:
    # if alg_name == 'RL':
    #     continue
    alg_results[alg_name].cal_rewards(norm_base=norm_base)
    # alg_results[alg_name].cal_ori_rewards(norm_base=norm_base)

############## plot figure #########

SMALL_SIZE = 8
MEDIUM_SIZE = 11
LARGE_SIZE = 14
LARGER_SIZE = 16
ADD_STDS = False

plt.rc('font', size=LARGE_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LARGE_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

index = np.arange(len(alg_names))
names = alg_names
labels = alg_names
color = ['#8437A6', '#4285F4', '#FBBC05', '#EA4335', '#34A853', '#FFDEAD']
error_kw = {'ecolor':'k', 'capsize':3, 'elinewidth':2}
patterns = ('|', '-', 'x', '//', '\\\\', '+')

plt.figure(figsize=(10,7))
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

# avgs[-1] = np.mean(our_reward)
# stds[-1] = np.std(our_reward)
print('Reward')
print(avgs)
print(stds)

# plt.title('Inference Latency')
plt.ylabel('Normalized average QoE')
# plt.bar(index, avgs, yerr=stds, color=color, error_kw=error_kw, label=labels, hatch=patterns)
hatch_density = 2
if ADD_STDS:
    for i, pattern in enumerate(patterns[:len(alg_names)]):
        plt.bar(i, avgs[i], hatch=pattern * hatch_density,
                label=labels[i], color=color[i], yerr=stds[i], error_kw=error_kw, width=1.0)
else:
    for i, pattern in enumerate(patterns[:len(alg_names)]):
        plt.bar(i, avgs[i], hatch=pattern * hatch_density,
                label=labels[i], color=color[i], error_kw=error_kw, width=1.0)

plt.xticks(index, names)
# plt.legend()
# plt.legend(bbox_to_anchor=(0.9, 1.10), ncol=len(alg_names), fontsize=12)
plt.legend(loc='lower left', bbox_to_anchor=(0., 0.98, 1., .08), mode='expand', ncol=len(alg_names), fontsize=13, frameon=False)


img_path = os.path.join(root, 'Normalized_average_QoE.jpg')
# show = True
# if show:
#     plt.show()
# else:
#     # plt.savefig('Normalized average QoE.eps', format='eps')
#     plt.savefig(img_path)
# plt.show()
plt.savefig(img_path)
plt.close()