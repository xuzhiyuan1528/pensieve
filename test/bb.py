import numpy as np
import fixed_env as env
import load_trace
import os


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
TRACE_DIR = './cooked_traces/'
SUMMARY_DIR = './gen-logs'
TRANS_DIR = './gen-traces'
LOG_FILE = SUMMARY_DIR + '/log_sim_bb'
TRANS_FILE = TRANS_DIR + '/trace_sim_bb'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

repeat_time = 1

# video chunk sizes
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

np.set_printoptions(linewidth=10000)

def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]

def main(log_suffix):
    run_id = '0'
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    if not os.path.exists(TRANS_DIR):
        os.makedirs(TRANS_DIR)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRACE_DIR)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx] + '_' + str(run_id)
    log_file = open(log_path, 'wb')
    trans_path = TRANS_FILE + '_' + all_file_names[net_env.trace_idx] + '_' + str(run_id)
    trans_file = open(trans_path, 'wb')

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    rl_batch = [np.zeros((S_INFO, S_LEN))]
    r_batch = []

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        # retrieve previous state
        if len(rl_batch) == 0:
            rl_state = [np.zeros((S_INFO, S_LEN))]
            old_rl_state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
            # print("length of rl_batch is 0")
        else:
            rl_state = np.array(rl_batch[-1], copy=True)
            old_rl_state = np.array(rl_batch[-1], copy=True)
            # print("length of rl_batch is not 0")

        # compute bandwidth measurement
        # video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
        video_chunk_fetch_time = delay

        video_chunk_coount = TOTAL_VIDEO_CHUNKS - video_chunk_remain

        # dequeue history record
        rl_state = np.roll(rl_state, -1, axis=1)

        # next_video_chunk_sizes = []
        # for i in xrange(A_DIM):
        #     next_video_chunk_sizes.append(get_chunk_size(i, video_chunk_coount))

        # generate state for RL
        try:
            rl_state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
            rl_state[1, -1] = buffer_size/ BUFFER_NORM_FACTOR
            rl_state[2, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
            rl_state[3, -1] = float(video_chunk_fetch_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            rl_state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            rl_state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if len(rl_batch) == 0:
                rl_state = [np.zeros((S_INFO, S_LEN))]
            else:
                rl_state = np.array(rl_batch[-1], copy=True)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)

        action_prob = np.zeros((len(VIDEO_BIT_RATE)), dtype=np.float64)
        action_prob[bit_rate] = 1.0

        trans_file.write('|'.join([str(list(old_rl_state.reshape(-1))),
                                      str(list(action_prob.reshape(-1))),
                                      str(list(rl_state.reshape(-1))),
                                      str(reward), str(bit_rate)]))
        trans_file.write('\n')
        trans_file.flush()

        if end_of_video:
            rl_batch = [np.zeros((S_INFO, S_LEN))]
        else:
            rl_batch.append(rl_state)

        if end_of_video:
            trans_file.write('\n')
            trans_file.close()

            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print "video count", video_count
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx] + '_' + str(run_id)
            log_file = open(log_path, 'wb')
            trans_path = TRANS_FILE + '_' + all_file_names[net_env.trace_idx] + '_' + str(run_id)
            trans_file = open(trans_path, 'wb')


if __name__ == '__main__':
    for i in range(repeat_time):
        main(i)
