import os
import numpy as np
import matplotlib.pyplot as plt

root = '/home/cst/Dropbox/0-DRL-Imitation/sim_results'
# base_dir = os.path.join(root, 'gen-traces')
# ori_base_dir = os.path.join(root, 'gen-logs')
RESULTS_FOLDER = os.path.join(root, 'gen-logs')
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet  # nipy_spectral, Set1,Paired
SIM_DP = 'sim_dp'
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
# SCHEMES = ['BOLA', 'RL', 'fastMPC', 'robustMPC', 'Ours', 'BB']
# SCHEMES = ['Ours', 'RL', 'fastMPC']
# SCHEMES = ['sim_rl', 'sim_rl003', 'sim_rl001', 'sim_rl005', 'sim_iml']
# SCHEMES = ['sim_rl', 'sim_mpc', 'sim_mpc001', 'sim_mpc003', 'sim_mpc005', 'sim_iml']

SCHEMES_CDF_NAME = 'Normalized_CDF.jpg'
SCHEMES_QoE_NAME = 'Normalized_average_QoE.jpg'
SCHEMES = ['sim_rl', 'sim_iml05', 'sim_mpc', 'sim_bb', 'sim_bcq05']
SCHEMES_suffix = '05_'
# SCHEMES = ['sim_rl', 'sim_iml', 'sim_mpc', 'sim_bb', 'sim_bcq']
# SCHEMES_suffix = ''

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
        if log_file[-1] != '0':
            continue

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        # print log_file

        with open(os.path.join(RESULTS_FOLDER, log_file), 'rb') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                time_ms.append(float(parse[0]))
                bit_rate.append(int(parse[1]))
                buff.append(float(parse[2]))
                bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                reward.append(float(parse[6]))

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            name = scheme + '_'
            if name in log_file:
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

    small_len_count = 0
    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                schemes_check = False
                print('log file ' + str(l) + ' has len smaller the VIDEO_LEN')
                small_len_count += 1
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                # reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))
                ave_reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][1:VIDEO_LEN]))
    print('Total %d logs' % len(time_all[SCHEMES[0]]))
    print('%d logs are too small.' % small_len_count)

    Normalization_RL = True
    if Normalization_RL:
        norm_base = np.mean(reward_all['sim_rl'])
        for scheme in SCHEMES:
            reward_all[scheme] = reward_all[scheme] / norm_base

    mean_rewards = {}
    std_rewards = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        std_rewards[scheme] = np.std(reward_all[scheme])
        print('%s : %.4f' % (scheme, mean_rewards[scheme]))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for scheme in SCHEMES:
    #     ax.plot(reward_all[scheme])
    #
    # SCHEMES_REW = []
    # for scheme in SCHEMES:
    #     SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))
    #
    # colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    # for i, j in enumerate(ax.lines):
    #     j.set_color(colors[i])
    #
    # ax.legend(SCHEMES_REW, loc=4)
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
        # print(scheme)
        # print(len(reward_all[scheme]), reward_all[scheme])
        # print(np.mean(reward_all[scheme]))
        # values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        # cumulative = np.cumsum(values)
        # ax.plot(base[:-1], cumulative)

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

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme)

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW, loc=len(SCHEMES_REW))

    plt.ylabel('CDF')
    plt.xlabel('Ave QoE')
    img_path = os.path.join(root, SCHEMES_suffix + SCHEMES_CDF_NAME)
    plt.savefig(img_path)
    plt.close()
    # plt.show()

    if 'sim_iml' in reward_all:
        for f, r in zip(log_file_all, reward_all['sim_iml']):
            print f, r
    for k, v in reward_all.items():
        mean = np.mean(v)  # / np.mean(reward_all['RL'])
        print k, mean, np.std(v)
    # exit()

    ######## plot ave QoE results ########

    SMALL_SIZE = 8
    MEDIUM_SIZE = 11
    LARGE_SIZE = 14
    LARGER_SIZE = 16
    ADD_STDS = False

    plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=LARGE_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    index = np.arange(len(SCHEMES))
    names = SCHEMES
    labels = SCHEMES
    color = ['#8437A6', '#4285F4', '#FBBC05', '#EA4335', '#34A853', '#FFDEAD']
    error_kw = {'ecolor': 'k', 'capsize': 3, 'elinewidth': 2}
    patterns = ('|', '-', 'x', '//', '\\\\', '+')

    plt.figure(figsize=(10, 7))
    plt.grid(linestyle='--', linewidth=0.5, axis='y', dashes=(10, 5))
    ax = plt.gca()
    ax.set_axisbelow(True)
    # avgs = [avg_times_list_cpu[0], avg_times_list_gpu[0], avg_times_list_rr[0], avg_times_list_dqn[0], avg_times_list_dqnp[0]]
    # stds = [std_times_list_cpu[0], std_times_list_gpu[0], std_times_list_rr[0], std_times_list_dqn[0], std_times_list_dqnp[0]]
    # avgs = np.random.rand(5)
    # stds = np.random.rand(5)
    avgs = np.zeros(len(SCHEMES), dtype=float)
    stds = np.zeros(len(SCHEMES), dtype=float)
    for i, alg_name in enumerate(SCHEMES):
        avgs[i] = mean_rewards[alg_name]
        stds[i] = std_rewards[alg_name]

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
        for i, pattern in enumerate(patterns[:len(SCHEMES)]):
            plt.bar(i, avgs[i], hatch=pattern * hatch_density,
                    label=labels[i], color=color[i], yerr=stds[i], error_kw=error_kw, width=1.0)
    else:
        for i, pattern in enumerate(patterns[:len(SCHEMES)]):
            plt.bar(i, avgs[i], hatch=pattern * hatch_density,
                    label=labels[i], color=color[i], error_kw=error_kw, width=1.0)

    plt.xticks(index, names)
    # plt.legend()
    # plt.legend(bbox_to_anchor=(0.9, 1.10), ncol=len(alg_names), fontsize=12)
    plt.legend(loc='lower left', bbox_to_anchor=(0., 0.98, 1., .08), mode='expand', ncol=len(SCHEMES), fontsize=13,
               frameon=False)

    img_path = os.path.join(root, SCHEMES_suffix + SCHEMES_QoE_NAME)
    show = True
    # if show:
    #     plt.show()
    # else:
    #     # plt.savefig('Normalized average QoE.eps', format='eps')
    #     plt.savefig(img_path)
    # plt.show()
    plt.savefig(img_path)
    plt.close()

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
    #             SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))
    #
    #         ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
    #         plt.show()
    #     break


if __name__ == '__main__':
    main()
