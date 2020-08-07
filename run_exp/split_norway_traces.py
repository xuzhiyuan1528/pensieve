import os
import random
import shutil

ori_results_folder = './results_ori/'
new_results_folder = './results/'

log_files = os.listdir(ori_results_folder)

num_test_traces = 20

trace_names = []
for log_file in log_files:
    start_index = log_file.find('BOLA')
    if start_index >= 0:
        trace_names.append(log_file[start_index+5:-2])

trace_names.sort(key=lambda x: os.path.getmtime(os.path.join(ori_results_folder, 'log_BOLA_' + x + '_0')))
print(trace_names)
print('################')

test_trace_names = random.sample(trace_names[:-2], k=num_test_traces)
test_trace_names = sorted(test_trace_names)

print('Choose ' + str(len(test_trace_names)) + ' traces')
for test_trace_name in test_trace_names:
    print(test_trace_name)

# Copy trace files
norway_train_trace_folder = '../norway_part_train_traces/'
norway_test_trace_folder = '../norway_part_test_traces/'
norway_trace_folder = '../norway_part_traces'

if not os.path.exists(norway_train_trace_folder):
    os.makedirs(norway_train_trace_folder)

if not os.path.exists(norway_test_trace_folder):
    os.makedirs(norway_test_trace_folder)

count = 0
for trace_name in os.listdir(norway_trace_folder):
    if trace_name in test_trace_names:
        src_file_path = os.path.join(norway_trace_folder, trace_name)
        tar_file_path = os.path.join(norway_test_trace_folder, trace_name)
        print('Test: Copy ' + str(src_file_path) + ' to ' + str(tar_file_path))
        count += 1
        shutil.copyfile(src_file_path, tar_file_path)
    else:
        src_file_path = os.path.join(norway_trace_folder, trace_name)
        tar_file_path = os.path.join(norway_train_trace_folder, trace_name)
        print('Train: Copy ' + str(src_file_path) + ' to ' + str(tar_file_path))
        shutil.copyfile(src_file_path, tar_file_path)

print('Copy ' + str(count) + ' files to test folder')

# Copy log files
test_results_folder = './test_results'
if not os.path.exists(test_results_folder):
    os.makedirs(test_results_folder)

test_ori_results_folder = os.path.join(test_results_folder, 'results_ori')
print(test_ori_results_folder)
if not os.path.exists(test_ori_results_folder):
    os.makedirs(test_ori_results_folder)

test_new_results_folder = os.path.join(test_results_folder, 'results')
print(test_new_results_folder)
if not os.path.exists(test_new_results_folder):
    os.makedirs(test_new_results_folder)

count = 0
for trace_name in test_trace_names:
    trace_log_count = 0
    for log_file in log_files:
        if log_file.find(trace_name+'_') >= 0:
            count += 1
            trace_log_count += 1
            src_log_path = os.path.join(ori_results_folder, log_file)
            tar_log_path = os.path.join(test_ori_results_folder, log_file)
            # print('################')
            # print('Copy ' + str(src_log_path) + ' to ' + str(tar_log_path))
            shutil.move(src_log_path, tar_log_path)

            src_log_path = os.path.join(new_results_folder, log_file)
            tar_log_path = os.path.join(test_new_results_folder, log_file)
            # print('Copy ' + str(src_log_path) + ' to ' + str(tar_log_path))
            shutil.move(src_log_path, tar_log_path)

    print('Copy ' + str(trace_log_count) + ' ' + str(trace_name) + ' log files')

print('Copy ' + str(count) + ' log files')



