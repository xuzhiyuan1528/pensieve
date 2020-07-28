#!/usr/bin/env python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import base64
import urllib
import sys
import os
import logging
import json

from collections import deque
import numpy as np
import time


S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0


VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TOTAL_VIDEO_CHUNKS = 48
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward

past_bandwidth_ests = []

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

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']

            self.rl_batch = input_dict['rl_batch']
            self.reward_batch = input_dict['reward_batch']

            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            
            print post_data
            send_data = ""

            # if ( 'lastquality' in post_data ):
            #     rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
            #     reward = \
            #        VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
            #        - REBUF_PENALTY * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K \
            #        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
            #                                       self.input_dict['last_bit_rate']) / M_IN_K
            #     # reward = BITRATE_REWARD[post_data['lastquality']] \
            #     #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])
            #
            #     video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
            #     video_chunk_size = post_data['lastChunkSize']
            #
            #     # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
            #     # self.log_file.write(str(time.time()) + '\t' +
            #     #                     str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
            #     #                     str(post_data['buffer']) + '\t' +
            #     #                     str(float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K) + '\t' +
            #     #                     str(video_chunk_size) + '\t' +
            #     #                     str(video_chunk_fetch_time) + '\t' +
            #     #                     str(reward) + '\n')
            #     # self.log_file.flush()
            #
            #     self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
            #     self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
            #
            #     if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
            #         send_data = "REFRESH"
            #         self.input_dict['last_total_rebuf'] = 0
            #         self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
            #         self.log_file.write('\n')  # so that in the log we know where video ends


            if ('lastquality' in post_data):
                rebuffer_time = float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])

                if len(self.reward_batch) == 0:
                    old_reward = 0.0
                    print("length of reward_batch is 0")
                else:
                    old_reward = np.array(self.reward_batch[-1], copy=True)
                    print("length of reward_batch is not 0")

                # --linear reward--
                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                         - REBUF_PENALTY * rebuffer_time / M_IN_K \
                         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                   self.input_dict['last_bit_rate']) / M_IN_K

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                # retrieve previous state
                if len(self.rl_batch) == 0:
                    rl_state = [np.zeros((S_INFO, S_LEN))]
                    old_rl_state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
                    last_old_rl_state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
                    print("length of rl_batch is 0")
                elif len(self.rl_batch) == 1:
                    rl_state = np.array(self.rl_batch[-1], copy=True)
                    old_rl_state = np.array(self.rl_batch[-1], copy=True)
                    last_old_rl_state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
                    print("length of rl_batch is 1")
                else:
                    rl_state = np.array(self.rl_batch[-1], copy=True)
                    old_rl_state = np.array(self.rl_batch[-1], copy=True)
                    last_old_rl_state = np.array(self.rl_batch[-2], copy=True)
                    print("length of rl_batch is more than 1")

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
                self.input_dict['video_chunk_coount'] += 1

                # dequeue history record
                rl_state = np.roll(rl_state, -1, axis=1)

                next_video_chunk_sizes = []
                for i in xrange(A_DIM):
                    next_video_chunk_sizes.append(get_chunk_size(i, self.input_dict['video_chunk_coount']))

                # generate state for RL
                try:
                    rl_state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
                    rl_state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                    rl_state[2, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                    rl_state[3, -1] = float(video_chunk_fetch_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    rl_state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    rl_state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
                        CHUNK_TIL_VIDEO_END_CAP)
                except ZeroDivisionError:
                    # this should occur VERY rarely (1 out of 3000), should be a dash issue
                    # in this case we ignore the observation and roll back to an eariler one
                    if len(self.rl_batch) == 0:
                        rl_state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        rl_state = np.array(self.rl_batch[-1], copy=True)

                last_action = post_data['lastquality']
                last_action_prob = np.zeros((len(VIDEO_BIT_RATE)), dtype=np.float64)
                last_action_prob[int(last_action)] = 1.0

                self.log_file.write('|'.join([str(list(last_old_rl_state.reshape(-1))),
                                              str(list(last_action_prob.reshape(-1))),
                                              str(list(old_rl_state.reshape(-1))),
                                              str(old_reward), str(last_action)]))
                self.log_file.write('\n')
                self.log_file.flush()

                end_of_video = False
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = "REFRESH"
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_coount'] = 0
                    self.log_file.write('\n')  # so that in the log we know where video ends

                if end_of_video:
                    self.rl_batch = [np.zeros((S_INFO, S_LEN))]
                    self.reward_batch = []
                else:
                    # print(rl_state.shape)
                    self.rl_batch.append(rl_state)
                    self.reward_batch.append(reward)

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(send_data))
            self.send_header('Access-Control-Allow-Origin', "*")
            self.end_headers()
            self.wfile.write(send_data)

        def do_GET(self):
            print >> sys.stderr, 'GOT REQ'
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    with open(log_file_path, 'wb') as log_file:
        rl_batch = [np.zeros((S_INFO, S_LEN))]
        reward_batch = []

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0

        video_chunk_count = 0

        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_coount': video_chunk_count,
                      'rl_batch': rl_batch,
                      'reward_batch': reward_batch}

        handler_class = make_request_handler(input_dict=input_dict)

        server_address = ('localhost', port)
        httpd = server_class(server_address, handler_class)
        print 'Listening on port ' + str(port)
        httpd.serve_forever()


def main():
    if len(sys.argv) >= 3:
        abr_algo = sys.argv[1]
        trace_file = sys.argv[2]
        run(log_file_path=LOG_FILE + '_' + abr_algo + '_' + trace_file)
    else:
        run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.debug("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
