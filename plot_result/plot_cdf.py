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

# root = '/home/cst/Dropbox/0-DRL-Imitation/norway_results/train_results'
root = '/home/cst/Dropbox/0-DRL-Imitation/norway_results/test_results'
# root = '/home/cst/Dropbox/0-DRL-Imitation/link304-results'
# root = '/home/cst/Dropbox/0-DRL-Imitation/sim_results'
RESULTS_FOLDER = os.path.join(root, 'results_ori')

NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48  # 64
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]
COLOR_MAP = plt.cm.jet  # nipy_spectral, Set1,Paired
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
SCHEMES = ['fastMPC', 'robustMPC', 'BOLA', 'RL', 'BB', 'Ours']
if root.find('link304') >= 0:
    SCHEMES = ['fastMPC', 'robustMPC', 'BOLA', 'RL', 'Ours']
SCHEMES_IMG_NAME = 'Normalized_CDF.jpg'

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
    ave_reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []
        ave_reward_all[scheme] = []

    # reward_all['Ours'] = our_reward
    small_len_count = 0
    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            # if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
            if l not in time_all[scheme]:
                schemes_check = False
                print('log file ' + str(l) + ' has len smaller the VIDEO_LEN')
                small_len_count += 1
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))
                ave_reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][1:VIDEO_LEN]))
                # reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:]))
                # reward_all[scheme].extend(raw_reward_all[scheme][l][1:])
    print('Total %d logs' % len(time_all[SCHEMES[0]]))
    print('%d logs are too small.' % small_len_count)


    mean_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        print('%s : %.4f' % (scheme, mean_rewards[scheme]))

    # mean_rewards['Ours'] = np.mean(reward_all['Ours'])

    # plot total reward
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # for scheme in SCHEMES:
    #     ax.plot(reward_all[scheme])
    # ax.plot(reward_all['Ours'])

    # SCHEMES_REW = []
    # for scheme in SCHEMES:
    #     SCHEMES_REW.append(scheme + ': %.4f' % mean_rewards[scheme])
    # SCHEMES_REW.append('Ours: %.4f' % mean_rewards['Ours'])

    # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # for i, j in enumerate(ax.lines):
    #     j.set_color(colors[i])
    #
    # ax.legend(SCHEMES_REW, loc=len(SCHEMES_REW))
    #
    # plt.ylabel('total reward')
    # plt.xlabel('trace index')
    # plt.show()

    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        # cur_reward_all = np.array(reward_all[scheme])
        cur_reward_all = np.array(ave_reward_all[scheme])
        # max_reward = np.max(cur_reward_all)
        # min_reward = np.min(cur_reward_all)
        # cur_reward_all = (cur_reward_all - min_reward) / (max_reward - min_reward)

        values, base = np.histogram(cur_reward_all, bins=NUM_BINS)
        cumulative = np.cumsum(values)
        cumulative = cumulative.astype(float)
        max_cumu = np.max(cumulative)
        min_cumu = np.min(cumulative)
        cur_cumu_all = (cumulative - min_cumu) / (max_cumu - min_cumu)
        ax.plot(base[:-1], cur_cumu_all)
    # values, base = np.histogram(reward_all['Ours'], bins=NUM_BINS)
    # cumulative = np.cumsum(values)
    # ax.plot(base[:-1], cumulative)

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme)

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=len(SCHEMES_REW))

    plt.ylabel('CDF')
    plt.xlabel('Ave QoE')
    img_path = os.path.join(root, SCHEMES_IMG_NAME)
    plt.savefig(img_path)
    plt.close()
    # plt.show()

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