import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 240  # sec
MM_DELAY = 40   # millisec

ABR_ALGO = ['Ours']
# ABR_ALGO = ['BOLA', 'fastMPC', 'RL', 'robustMPC']

os.system('sudo sysctl -w net.ipv4.ip_forward=1')

TRACE_PATH = '../pantheon_traces/'
# TRACE_PATH = '../norway_part_traces/'

ip = '100.64.0.1'
REPEAT_TIME = 20

def main():
	# trace_path = sys.argv[1]
	# abr_algo = sys.argv[2]
	# process_id = sys.argv[3]
	# ip = sys.argv[4]
	trace_path = TRACE_PATH

	sleep_vec = range(2, 10)  # random sleep second

	files = os.listdir(trace_path)

	for rt in xrange(REPEAT_TIME):

		for f in files:
			for process_id, abr_algo in enumerate(ABR_ALGO):

				process_id = str(process_id)

			# for f in files:

				# if f.find('77.72mbps') < 0:
				# 	continue

				if f.find('3.04mbps-poisson') < 0:
					continue

				while True:

					np.random.shuffle(sleep_vec)
					sleep_time = sleep_vec[int(process_id)]

					# # 77.72mbps
					# script = 'mm-delay 51' + ' mm-loss uplink 0.0006' + \
					# 		 ' mm-link ' + trace_path + f + ' ' + trace_path + f + \
					# 		 ' --uplink-queue=droptail --uplink-queue-args=packets=94' + \
					# 		 ' /usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' + \
					# 		 abr_algo + ' ' + str(RUN_TIME) + ' ' + \
					# 		 process_id + ' ' + f + ' ' + str(sleep_time) + ' ' + str(rt)
					# print(script)

					# 3.04
					script = 'mm-delay 130' + ' mm-link ' + trace_path + f + ' ' + trace_path + f + \
							 ' --uplink-queue=droptail --uplink-queue-args=packets=426' + \
							 ' /usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' + \
							 abr_algo + ' ' + str(RUN_TIME) + ' ' + \
							 process_id + ' ' + f + ' ' + str(sleep_time) + ' ' + str(rt)
					print(script)

					# '12mbps.trace'
					# script = 'mm-delay 30' + ' mm-link ' + trace_path + f + ' ' + trace_path + f + \
					# 		 ' --uplink-queue=droptail --uplink-queue-args=bytes=90000' + \
					# 		 ' /usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' + \
					# 		 abr_algo + ' ' + str(RUN_TIME) + ' ' + \
					# 		 process_id + ' ' + f + ' ' + str(sleep_time) + ' ' + str(rt)
					# print(script)

					# proc = subprocess.Popen(script, shell=True)

					# script = 'mm-delay ' + str(MM_DELAY) + \
					# 		 ' mm-link 12mbps ' + trace_path + f + ' ' + \
					# 		 '/usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' + \
					# 		 abr_algo + ' ' + str(RUN_TIME) + ' ' + \
					# 		 process_id + ' ' + f + ' ' + str(sleep_time) + ' ' + str(rt),
					# print(script)

					# proc = subprocess.Popen(script, shell=True)
					proc = subprocess.Popen(script,
											stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

					# proc = subprocess.Popen('mm-delay ' + str(MM_DELAY) +
					# 		  ' mm-link 12mbps ' + trace_path + f + ' ' +
					# 		  '/usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' +
					# 		  abr_algo + ' ' + str(RUN_TIME) + ' ' +
					# 		  process_id + ' ' + f + ' ' + str(sleep_time),
					# 		  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

					# script = 'mm-delay ' + str(MM_DELAY) + \
					# 		 ' mm-link 12mbps ' + trace_path + f + ' '
					# print(script)
					# proc = subprocess.Popen('mm-delay ' + str(MM_DELAY) +
					# 						' mm-link 12mbps ' + trace_path + f + ' ' +
					# 						'/usr/bin/python ' + RUN_SCRIPT + ' ' + ip + ' ' +
					# 						abr_algo + ' ' + str(RUN_TIME) + ' ' +
					# 						process_id + ' ' + f + ' ' + str(sleep_time),
					# 						shell=True)

					(out, err) = proc.communicate()

					if out is None:
						out = " none"
					print(out)

					if out.find('done') >= 0:
						break
					else:
						with open('./chrome_retry_log', 'ab') as log:
							log.write(abr_algo + '_' + f + '\n')
							log.write(out + '\n')
							log.flush()



if __name__ == '__main__':
	main()
