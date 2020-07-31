import os
import time
import json
import urllib
import subprocess


# TRACE_PATH = '../cooked_traces/'
TRACE_PATH = '../pantheon_traces/'

with open('./chrome_retry_log', 'wb') as f:
	f.write('chrome retry log\n')

os.system('sudo sysctl -w net.ipv4.ip_forward=1')

# ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
# ip = str(ip_data['ip'])
# ip = '127.0.0.1'
# ip = 'localhost'
ip = '192.168.0.136'
print(ip)

ABR_ALGO = 'BB'
PROCESS_ID = 0
command_BB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_BB)

ABR_ALGO = 'RB'
PROCESS_ID = 1
command_RB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_RB)

ABR_ALGO = 'FIXED'
PROCESS_ID = 2
command_FIXED = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_FIXED)

ABR_ALGO = 'FESTIVE'
PROCESS_ID = 3
command_FESTIVE = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_FESTIVE)

ABR_ALGO = 'BOLA'
PROCESS_ID = 4
command_BOLA = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_BOLA)

ABR_ALGO = 'fastMPC'
PROCESS_ID = 5
command_fastMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_fastMPC)

ABR_ALGO = 'robustMPC'
PROCESS_ID = 6
command_robustMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_robustMPC)

ABR_ALGO = 'RL'
PROCESS_ID = 7
command_RL = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip
print(command_RL)

# proc_BB = subprocess.Popen(command_BB, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# proc_RB = subprocess.Popen(command_RB, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# proc_FIXED = subprocess.Popen(command_FIXED, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
# proc_FESTIVE = subprocess.Popen(command_FESTIVE, stdout=subprocess.PIPE, shell=True)
# time.sleep(0.1)
proc_BOLA = subprocess.Popen(command_BOLA, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
(proc_BOLA_out, _) = proc_BOLA.communicate()
print(proc_BOLA_out)

proc_fastMPC = subprocess.Popen(command_fastMPC, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
(proc_fastMPC_out, _) = proc_fastMPC.communicate()
print(proc_fastMPC_out)

proc_robustMPC = subprocess.Popen(command_robustMPC, stdout=subprocess.PIPE, shell=True)
# proc_robustMPC = subprocess.Popen(command_robustMPC, shell=True)
time.sleep(0.1)
(proc_robustMPC_out, _) = proc_robustMPC.communicate()
print(proc_robustMPC_out)

proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)
# proc_RL = subprocess.Popen(command_RL, shell=True)
time.sleep(0.1)
(proc_RL_out, _) = proc_RL.communicate()
print(proc_RL_out)

# proc_BB.wait()
# proc_RB.wait()
# proc_FIXED.wait()
# proc_FESTIVE.wait()
proc_BOLA.wait()
proc_fastMPC.wait()
proc_robustMPC.wait()
proc_RL.wait()
