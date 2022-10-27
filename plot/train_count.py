import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'font.family': 'Myriad Pro'})
number_of_retrain = 20
def train_count_first(batch_num, replay_ratio, phase1_count, phase2_count):
    training_counts = np.zeros(batch_num)
    # when the no batch comes
    for no in np.arange(batch_num):
        left_num = 0
        # start from highest priority
        for i in np.arange(no + 1)[::-1]:
            if i == no:
                training_counts[no] = 1
            else:
                sample_sum = phase1_count + phase2_count * (no - 1)
                retrain_count_sum = np.sum(np.square(np.arange(0, no + 1)))
                retrain_count = np.square(i + 1)
                if i == 0:
                    current_batch_num = phase1_count
                else:
                    current_batch_num = phase2_count
                replay_sample_num = replay_ratio * sample_sum
                if left_num == 0 and replay_sample_num * (retrain_count / retrain_count_sum) > current_batch_num:
                    count = 1
                    left_num = replay_sample_num - current_batch_num
                elif left_num == 0:
                    count = replay_ratio * (sample_sum / current_batch_num) * (retrain_count / retrain_count_sum)
                else:
                    if left_num != 0 and current_batch_num < left_num:
                        count = 1
                        left_num = left_num - current_batch_num
                    else:
                        count = left_num / current_batch_num
                training_counts[i] += count

    return training_counts

def train_count(batch_num, replay_ratio, phase1_count, phase2_count):
    training_counts = np.zeros(batch_num)
    # iterate over batch
    for no in np.arange(batch_num):
        # from current batch to the end, sum all the training times
        for i in np.arange(no, batch_num):
            if i == 0:
                training_counts[no] = 1
            else:
                sample_sum = phase1_count + phase2_count * (i - 1)
                retrain_count_sum = np.sum(np.arange(1, i + 1))
                retrain_count = no + 1
                if no == 0:
                    current_batch_num = phase1_count
                else:
                    current_batch_num = phase2_count
                count = replay_ratio * (sample_sum / current_batch_num) * (retrain_count / retrain_count_sum)
                training_counts[no] += count
    
    return training_counts

training_counts = train_count_first(4, 0.5, 100, 100)
print(training_counts)
# WIKI
phase1_ratio = 0.3
retrain_intervals = [50000, 30000, 20000, 10000, 5000]
edges = {
    'wiki': 157000,
    'mooc': 412000,
    'reddit': 672000,
}
linewidth = 5
for (data, edge) in edges.items():
    plt.figure(figsize=(12, 9))
    plt.title(data)
    for replay in [1, 0.5, 0.2, 0]:
        for retrain_interval in retrain_intervals:
            phase1 = phase1_ratio * edge
            phase2_num = (1 - phase1_ratio) * edge
            batch_num = int(phase2_num // retrain_interval) + 1
            training_counts = train_count_first(batch_num, replay, phase1, retrain_interval)
            print("replay: {}".format(replay))
            print("retrain_interval: {}".format(retrain_interval))
            print(training_counts)
            plt.plot(np.arange(1, 1 + len(training_counts)), training_counts, linewidth=linewidth, label="replay = {}, retrain interval = {}".format(replay, retrain_interval))
    plt.xlabel('No.# batch in online learning')
    plt.ylabel('training times')
    plt.legend()
    plt.savefig("{}_training_count.png".format(data), dpi=400, bbox_inches='tight')
