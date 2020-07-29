import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 320 # sec
# ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
# ABR_ALGO = ['fastMPC', 'robustMPC', 'RL']
# ABR_ALGO = ['Ours']
ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA']
# ABR_ALGO = ['BOLA']
REPEAT_TIME = 1000

terminal_index = 0
if len(sys.argv) >= 2:
	terminal_index = sys.argv[1]

def main():

	np.random.seed(RANDOM_SEED)

	with open('./chrome_retry_log_'+str(terminal_index), 'wb') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in xrange(REPEAT_TIME):
			np.random.shuffle(ABR_ALGO)
			for abr_algo in ABR_ALGO:

				while True:

					script = 'python ' + RUN_SCRIPT + ' ' + \
							  abr_algo + ' ' + str(RUN_TIME) + \
							 ' ' + str(rt) + ' ' + str(terminal_index)
					
					proc = subprocess.Popen(script,
							  stdout=subprocess.PIPE,
							  stderr=subprocess.PIPE,
							  shell=True)

					# proc = subprocess.Popen(script,
					# 		  shell=True)

					(out, err) = proc.communicate()
					print(out)

					if out.find('done') >= 0:
						break
					else:
						log.write(abr_algo + '_' + str(rt) + '\n')
						log.write(out + '\n')
						log.flush()



if __name__ == '__main__':
	main()
