import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.model_selection import StratifiedKFold



def build_customized_feature_matrix(feat_file_lst):

    feat_dic = pd.read_csv(feat_file_lst, sep='\t', index_col=0)
    feat_raw = feat_dic

    return pd.DataFrame(feat_raw, index=feat_dic.index)

def generate_CPDB_5CV_set(drivers,nondrivers,randseed):
    X1, y1 = drivers, [1]*len(drivers)
    X2, y2 = nondrivers, [0] * len(nondrivers)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=randseed)
    X_5CV = {}
    cv_idx=1

    for train1, test1 in skf.split(X1, y1):
        train = train1
        test = test1

        train_set1 = []
        train_label1 = []
        test_set1 = []
        test_label1 = []
        for i in train:
            train_set1.append(X1[i])
            train_label1.append(y1[i])
        for i in test:
            test_set1.append(X1[i])
            test_label1.append(y1[i])

        X_5CV['train_%d' % cv_idx] = train_set1
        X_5CV['test_%d' % cv_idx] = test_set1
        X_5CV['train_label_%d' % cv_idx] = train_label1
        X_5CV['test_label_%d' % cv_idx] = test_label1
        cv_idx = cv_idx + 1

    cv_idx=1
    for train2, test2 in skf.split(X2, y2):
        train = train2
        test = test2

        train_set2 = []
        train_label2 = []
        test_set2 = []
        test_label2 = []
        for i in train:
            train_set2.append(X2[i])
            train_label2.append(y2[i])
        for i in test:
            test_set2.append(X2[i])
            test_label2.append(y2[i])

        X_5CV['train_%d' % cv_idx] += train_set2
        X_5CV['test_%d' % cv_idx] += test_set2
        X_5CV['train_label_%d' % cv_idx] += train_label2
        X_5CV['test_label_%d' % cv_idx] += test_label2
        cv_idx = cv_idx + 1

    return X_5CV
def load_obj( name ):
    with open( name , 'rb') as f:
        return pickle.load(f)
#Loading data
datafile = "BioPlex"
if datafile == "CPDB":
    network_file = './data/CPDB/CPDB_edgelist.txt'
    feat_file_lst = "./data/CPDB/CPDB_omics_features_data_UCEC.tsv"
    savepath = "./data/CPDB/CPDB_12850_5CV_UCEC.pkl"
    # ppifile = "./data/CPDB/CPDB_PPI_data.tsv"
    ppiAdj = load_obj("./data/CPDB/CPDB_ppi_3UCEC.pkl")
    
elif datafile == "BioPlex":
    network_file = './data/BioPlex/BioPlex_edge_file.txt'
    feat_file_lst = "./data/BioPlex/BioPlex_omics_features_data_UCEC.tsv"
    savepath = "./data/BioPlex/BioPlex_8304_5CV_UCEC.pkl"
    # ppifile = "./data/STRING/STRING_PPI_data.tsv"
    ppiAdj = load_obj("./data/BioPlex/BioPlex_ppi_3UCEC.pkl")

elif datafile == "irefindex":
    network_file = './data/irefindex/irefindex_edge_file.txt'
    feat_file_lst = "./data/irefindex/irefindex_omics_features_data_UCEC.tsv"
    savepath = "./data/irefindex/irefindex_11777_5CV_UCEC.pkl"
    ppiAdj = load_obj("./data/irefindex/irefindex_ppi_3UCEC.pkl")


# Concatenate multiple features to form one feature matrix
net_features = build_customized_feature_matrix(feat_file_lst)

dataset = dict()
dataset['feature'] = torch.FloatTensor(np.array(net_features))
dataset['node_name'] = net_features.index.values.tolist()
dataset['edge_index'] = ppiAdj["edge_index"]
dataset['feature_name'] = net_features.columns.values.tolist()

# d_lst = pd.read_table(filepath_or_buffer='./data/796_drivers.txt', sep='\t', header=None, index_col=None, names=['driver']) #Pan-Cancer
d_lst = pd.read_table(filepath_or_buffer='./data/cancer/pos-ucec.txt', sep='\t', header=None, index_col=None, names=['driver']) #Single cancer
d_lst = d_lst['driver'].values.tolist()

# nd_lst = pd.read_table(filepath_or_buffer='./data/2187_nondrivers.txt', sep='\t', header=None, index_col=None, names=['nondriver'])
nd_lst = pd.read_table(filepath_or_buffer='./data/cancer/neg.txt', sep='\t', header=None, index_col=None, names=['nondriver'])
nd_lst = nd_lst['nondriver'].values.tolist()

# True labels of genes
labels = []
mask = []
for g in dataset['node_name']:
    if g in d_lst:
        labels.append(1)
    else:
        labels.append(0)
    if (g in d_lst) or (g in nd_lst):
        mask.append(True)
    else:
        mask.append(False)

d_in_net = [] # Canonical driver genes in the network
nd_in_net = [] # Nondriver genes in the network
for g in dataset['node_name']:
    if g in d_lst:
        d_in_net.append(g)
    elif g in nd_lst:
        nd_in_net.append(g)

k_sets_net = dict()
for k in np.arange(0,10): # Randomly generate 5CV splits for ten times
    k_sets_net[k] = []
    randseed = (k+1)%100+(k+1)*5
    cv = generate_CPDB_5CV_set(d_in_net,nd_in_net,randseed)
    for cv_idx in np.arange(1,6):
        tr_mask = []
        te_mask = []
        for g in dataset['node_name']:
            if g in cv['train_%d' % cv_idx]:
                tr_mask.append(True)
            else:
                tr_mask.append(False)
            if g in cv['test_%d' % cv_idx]:
                te_mask.append(True)
            else:
                te_mask.append(False)
        tr_mask = np.array(tr_mask)
        te_mask = np.array(te_mask)
        k_sets_net[k].append((tr_mask,te_mask))


dataset['label'] = torch.FloatTensor(np.array(labels))
dataset['split_set'] = k_sets_net
dataset['mask'] = np.array(mask)

# Save the dataset as pickle file
with open(savepath, 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


