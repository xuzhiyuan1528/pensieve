import os
import numpy as np
import matplotlib.pyplot as plt

# root = '/home/cst/wk/Pensieve/data/norway_results/train_results/'
#
# # RESULTS_FOLDER = './results_ori/'
# # RESULTS_FOLDER = '/home/cst/wk/Pensieve/data/results_304mbps_20200801/'
# RESULTS_FOLDER = os.path.join(root, 'results_ori')
#
# our_result_dir = os.path.join(root, 'RL-best')
# our_state_path = os.path.join(our_result_dir, 'sample-state.npy')
# our_action_path = os.path.join(our_result_dir, 'sample-action.npy')
# our_reward_path = os.path.join(our_result_dir, 'sample-reward.npy')
# our_ndone_path = os.path.join(our_result_dir, 'sample-ndone.npy')
#
# our_state = np.load(our_state_path)
# our_action = np.load(our_action_path)
# our_reward = np.load(our_reward_path)
# our_ndone = np.load(our_ndone_path)

RESULTS_FOLDER = '/home/cst/Dropbox/0-DRL-Imitation/norway_results/train_results/results_ori'

NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48  # 64
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]
COLOR_MAP = plt.cm.jet  # nipy_spectral, Set1,Paired
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
SCHEMES = ['fastMPC', 'robustMPC', 'BOLA', 'RL', 'BB', 'Ours']

def main():
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        print log_file

        file_path = os.path.join(RESULTS_FOLDER, log_file)

        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                time_ms.append(float(parse[0]))
                bit_rate.append(int(parse[1]))
                buff.append(float(parse[2]))
                bw.append(float(parse[4]) / float(parse[5]) *
                          BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                reward.append(float(parse[6]))

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []

    # reward_all['Ours'] = our_reward

    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            # if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
            if l not in time_all[scheme]:
                schemes_check = False
                break
            if schemes_check:
                log_file_all.append(l)
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))
                # reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:]))
                # reward_all[scheme].extend(raw_reward_all[scheme][l][1:])

    mean_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        print('%s : %.4f' % (scheme, mean_rewards[scheme]))

    # mean_rewards['Ours'] = np.mean(reward_all['Ours'])

    # plot total reward
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        ax.plot(reward_all[scheme])
    # ax.plot(reward_all['Ours'])

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme + ': %.4f' % mean_rewards[scheme])
    # SCHEMES_REW.append('Ours: %.4f' % mean_rewards['Ours'])

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=len(SCHEMES_REW))

    plt.ylabel('total reward')
    plt.xlabel('trace index')
    plt.show()

    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        ax.plot(base[:-1], cumulative)
    # values, base = np.histogram(reward_all['Ours'], bins=NUM_BINS)
    # cumulative = np.cumsum(values)
    # ax.plot(base[:-1], cumulative)

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=len(SCHEMES_REW))

    plt.ylabel('CDF')
    plt.xlabel('total reward')
    plt.show()

    # ---- ---- ---- ----
    # check each trace
    # ---- ---- ---- ----

    # for l in time_all[SCHEMES[0]]:
    #     schemes_check = True
    #     for scheme in SCHEMES:
    #         if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
    #             schemes_check = False
    #             break
    #     if schemes_check:
    #         fig = plt.figure()
    #
    #         ax = fig.add_subplot(311)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.title(l)
    #         plt.ylabel('bit rate selection (kbps)')
    #
    #         ax = fig.add_subplot(312)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.ylabel('buffer size (sec)')
    #
    #         ax = fig.add_subplot(313)
    #         for scheme in SCHEMES:
    #             ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
    #         colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    #         for i, j in enumerate(ax.lines):
    #             j.set_color(colors[i])
    #         plt.ylabel('bandwidth (mbps)')
    #         plt.xlabel('time (sec)')
    #
    #         SCHEMES_REW = []
    #         for scheme in SCHEMES:
    #             if scheme == SIM_DP:
    #                 SCHEMES_REW.append(scheme + ': ' + str(raw_reward_all[scheme][l]))
    #             else:
    #                 SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))
    #
    #         ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
    #         plt.show()


if __name__ == '__main__':
    main()