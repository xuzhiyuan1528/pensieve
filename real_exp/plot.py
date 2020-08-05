import os
import numpy as np
import matplotlib.pyplot as plt

def deal_with_pensieve(fname):
    state = []
    action = []
    nxstate = []
    reward = []
    notdone = []
    with open(os.path.join(base_dir, fname), 'r') as fin:
        for row in fin:
            if len(row) < 10:
                continue
            tmp = row.split('|')
            # print(tmp)

            sep = ',' if ',' in tmp[0] else ' '
            state.append(np.fromstring(tmp[0][1:-2], dtype=np.float, sep=sep))

            sep = ',' if ',' in tmp[1] else ' '
            action.append(np.fromstring(tmp[1][1:-2], dtype=np.float, sep=sep))
            action[-1] = np.argmax(action[-1])

            sep = ',' if ',' in tmp[2] else ' '
            nxstate.append(np.fromstring(tmp[2].replace('[', '').replace(']', '').strip(), dtype=np.float, sep=sep))
            reward.append(float(tmp[3]))
            notdone.append(1 - (0 if 'False' in tmp[4] else 1))

    return state, action, reward, nxstate, notdone

# base_dir = '/home/eric/Dropbox/Projects-Research/0-DRL-Imitation/Pensieve_Tokyo_MPC_BOLA_320s'
# base_dir = '/home/cst/wk/Pensieve/data/results_0'
# base_dir = '/home/cst/wk/Pensieve/data/results_loss50'
# base_dir = '/home/cst/wk/Pensieve/pensieve/run_exp/results'
base_dir = '/home/cst/wk/Pensieve/data/results_304mbps_20200801'
# base_dir = '/home/cst/wk/Pensieve/data/results_7772mbps_20200802'

results = {
    'BOLA': [],
    'fastMPC': [],
    'robustMPC': [],
    'Our': [],
    'RL': []
}

datas = {
    'BOLA': [],
    'fastMPC': [],
    'robustMPC': [],
    'Our': [],
    'RL': []
}

for fname in os.listdir(base_dir):
    if 'log' not in fname:
        continue

    if fname.find('3.04mbps-poisson') < 0:
        continue

    # if fname.find('77.72mbps') < 0:
    #     continue

    algo_name  = fname.split('_')[1]

    state, action, reward, nxstate, notdone = deal_with_pensieve(fname)
    results[algo_name].append(np.mean(reward))
    # results[algo_name] = np.mean(reward)
    datas[algo_name] = reward

# print(f'reward {results}')

print('\n'.join("{}: {}".format(k, len(v)) for k, v in results.items()))
print('\n'.join("{}: {}".format(k, v) for k, v in results.items()))

alg = 'robustMPC'

reward = datas[alg]
# print(action)
plt.title(alg)
plt.plot(reward)
plt.show()