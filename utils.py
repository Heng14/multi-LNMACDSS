"""
@author: Yi Cui
"""

import numpy as np
import random


def generate_toy_datasets(num_datasets, n_min, n_max, dim, lam=10, prob=0.5):
    '''
    Generate toy datasets for testing
    Arg
        num_datasets: (int), number of datasets to generate
        n_min: (int), minimum number of samples in each dataset
        n_max: (int), maximum number of samples in each dataset
        dim: (int), feature dimension
        lam: (float), mean of exponential distribution to sample survival time
        prob: (float), probability of events
    Return:
        datasets: a list of (X, time, event) tuples
    '''
    datasets = []
    for _ in range(num_datasets):
        n = random.randint(n_min, n_max)
        X = np.random.randn(n, dim)
        time = np.random.exponential(lam, n)
        event = np.random.binomial(1, prob, n)
        datasets.append((X, time, event)) 
    return datasets


def train_test_split(datasets, test_size):
    '''
    Split datasets by stratified sampling
    Each dataset in datasets are equally split according to test_size
    Arg
        datasets: a list of (X, time, event) tuples
        test_size: (float) proportion of datasets assigned for test data
    Return
        datasets_train: a list of (X_train, time_train, event_train) tuples
        datasets_test: a list of (X_test, time_test, event_test) tuples
    '''
    datasets_train = [] 
    datasets_test = []
    for X, time, event in datasets:
        n = X.shape[0]
        idx = np.random.permutation(n)
        idx_train = idx[int(n*test_size):]
        idx_test = idx[:int(n*test_size)]
        datasets_train.append((X[idx_train], time[idx_train], event[idx_train]))
        datasets_test.append((X[idx_test], time[idx_test], event[idx_test]))
    return datasets_train, datasets_test


def combine_datasets(datasets):
    '''
    Combine all the datasets into a single dataset
    Arg
        datasets: datasets: a list of (X, time, event) tuples
    Return
        X: combined design matrix
        time: combined survival time
        event: combined event
    '''

    X, time, event = zip(*datasets)

    X = np.concatenate(X, axis=0)
    time = np.concatenate(time, axis=0)
    event = np.concatenate(event, axis=0)

    #print (X.shape)
    #print (time.shape)
    #raise
    return X, time, event

#def get_datasets(datasets):
#    '''
#    get X time event from single dataset
#    ''' 
    



def get_index_pairs(datasets):
    '''
    For each dataset in datasets, get index pairs (idx1,idx2) satisfying time[idx1]<time[idx2] and event[idx1]=1
    Arg
        datasets: a list of (X, time, event) tuples
    Return
        index_pairs: a list of (idx1, idx2) tuples, where idx1 and idx2 are index vectors of the same length
    '''
    index_pairs = []
    for _, time, event in datasets:
        index_pairs.append(np.nonzero(np.logical_and(np.expand_dims(time,-1)<time, np.expand_dims(event,-1))))
    return index_pairs


def batch_factory(X, time, event, batch_size):
    print (X.shape, time.shape, event.shape)
    n = X.shape[0]
    num_batches = n//batch_size
    idx = np.random.permutation(n)
    X, time, event = X[idx], time[idx], event[idx] # randomly shuffle data
    start = 0
    def next_batch():
        nonlocal start
        X_batch = X[start:start+batch_size]
        time_batch = time[start:start+batch_size]
        event_batch = event[start:start+batch_size]
        start = (start+batch_size)%n
        return X_batch, time_batch, event_batch
    return next_batch, num_batches

def write_socre(score_all, dataset_name, ignore_list, flag):

    file_f = open(f"{flag}_score.txt","w")
    assert len(score_all)==len(dataset_name), "len not equal!"
    for i, score_i in enumerate(score_all):
        if i not in ignore_list:
            file_f.write(dataset_name[i]+" "+str(score_i[0])+"\n")

    file_f.close()

def cal_ci(score_all, time_all, event_all, dataset_name, flag, writeout = True):

    record = np.zeros_like(score_all)

    ignore_list_train = [192, 681, 854, 264, 37, 89, 701, 147, 200, 231, 79, 197, 894, 252, 396, 753, 267, 456, 570, 792, 823, 729, 433, 38, 524, 749, 887, 96, 386, 864, 161, 719, 895, 224, 549, 805, 259, 180, 732, 470, 896, 210, 204, 821, 426, 836, 227, 223, 214, 929, 445, 71, 750, 841, 366, 311, 685, 736, 138, 24, 28, 807, 319, 165, 438, 174, 476, 477, 506, 213, 298, 40, 511, 405, 713, 740, 159, 407, 950, 754, 371, 711, 778, 95, 882, 773, 493, 380, 471, 881, 519, 423, 158, 518, 649, 188, 249, 838, 930, 562, 632, 163, 897, 788, 867, 164, 861, 914, 48, 559, 322, 69, 916, 600, 948, 413, 592, 951, 248, 17, 306, 797, 389, 650, 781, 793, 2, 544, 554, 313, 4, 911, 83, 121, 910, 130, 222, 440, 782, 358, 261, 277, 326, 403, 639, 378, 846, 181, 369, 508, 356, 952, 132]
    ignore_list_val = [101, 167, 202, 163, 224, 136, 141, 248, 61, 0, 56, 51, 8, 183, 18]#42, 256, 2, 98, 194, 
    ignore_list_test = [16, 472, 496, 83, 214, 490, 198, 233, 129, 377, 185, 193, 339, 455, 238, 329, 89, 184, 516, 318, 137, 275, 328, 62, 266, 25, 35, 227, 267, 375, 14, 387, 9, 176, 223, 23, 113, 347, 21, 228]

    assert len(score_all)==len(time_all)==len(event_all), "len not equal!!!"
    count_all = 0
    count_succ = 0
    count_fail = 0

    if flag == 'train':
        ignore_list = ignore_list_train
    elif flag == 'val':
        ignore_list = ignore_list_val
    elif flag == 'test':
        ignore_list = ignore_list_test
    else:
        raise

    for i, score_i in enumerate(score_all):
        for j, score_j in enumerate(score_all):
            if time_all[i]<time_all[j] and event_all[i]==1 and (i not in ignore_list) and (j not in ignore_list):
                count_all += 1 
                if score_all[i]<score_all[j]:
                    count_succ += 1
                else:
                    count_fail += 1
                    record[i] += 1
                    record[j] += 1

    print (flag, len(ignore_list_train))
    #print ("max and location: ",record.max(), np.argmax(record, axis=0))
    print (np.argsort(record, axis=0)[-20:])
    ci = count_succ/count_all
    print (f'cal_ci: {ci}')

    if writeout:
        write_socre(score_all, dataset_name, ignore_list, flag)


if __name__=='__main__':

	n_datasets = 10
	n_min, n_max = 20, 30
	n_features = 40
	datasets = generate_toy_datasets(n_datasets, n_min, n_max, n_features)
	index_pairs = get_index_pairs(datasets)
	for i, (_, time, event) in enumerate(datasets):
	    idx1, idx2 = index_pairs[i]
	    assert np.all(time[idx1]<time[idx2])
	    assert np.all(event[idx1])
