import os
import gc
import random
from collections import defaultdict
import numpy as np
import pandas as pd

from joblib import delayed, Parallel, dump, load

import networkx as nx
from sklearn.decomposition import NMF


NB_CORES = 10

train = pd.read_csv('./data/raw_data/train.csv')
test = pd.read_csv('./data/raw_data/test.csv')

if not os.path.exists("../cache/features_aug.jl" ):

    def create_question_hash(train_df, test_df):
        train_qs = np.dstack([train_df["qid1"], train_df["qid2"]]).flatten()
        test_qs = np.dstack([test_df["qid1"], test_df["qid2"]]).flatten()
        all_qs = np.append(train_qs, test_qs)
        all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
        all_qs.reset_index(inplace=True, drop=True)
        question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
        return question_dict


    def get_hash(df, hash_dict):
        df["q1"] = df["qid1"].map(hash_dict)
        df["q2"] = df["qid2"].map(hash_dict)
        return df.drop(["qid1", "qid2"], axis=1)


    def get_kcore_dict(df):
        g = nx.Graph()
        g.add_nodes_from(df.q1)
        edges = list(df[["q1", "q2"]].to_records(index=False))
        g.add_edges_from(edges)
        g.remove_edges_from(g.selfloop_edges())

        print(len(g.nodes()))

        df_output = pd.DataFrame(data=list(g.nodes()), columns=["qid"])
        df_output["kcore"] = 0
        for k in range(2, NB_CORES + 1):
            ck = nx.k_core(g, k=k).nodes()
            print("kcore", k)
            df_output.ix[df_output.qid.isin(ck), "kcore"] = k
        return df_output.to_dict()["kcore"]

    def get_kcore_features(df, kcore_dict):
        df["kcore1"] = df["q1"].apply(lambda x: kcore_dict[x])
        df["kcore2"] = df["q2"].apply(lambda x: kcore_dict[x])
        return df

    def convert_to_minmax(df, col):
        sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
        df["min_" + col] = sorted_features[:, 0]
        df["max_" + col] = sorted_features[:, 1]
        return df.drop([col + "1", col + "2"], axis=1)

    def get_neighbors(train_df, test_df):
        neighbors = defaultdict(set)
        for df in [train_df, test_df]:
            for q1, q2 in zip(df["q1"], df["q2"]):
                neighbors[q1].add(q2)
                neighbors[q2].add(q1)
        return neighbors


    def get_neighbor_features(df, neighbors):
        common_nc = df.apply(lambda x: len(neighbors[x.q1].intersection(neighbors[x.q2])), axis=1)
        min_nc = df.apply(lambda x: min(len(neighbors[x.q1]), len(neighbors[x.q2])), axis=1)
        df["common_neighbor_ratio"] = common_nc / min_nc
        df["common_neighbor_count"] = common_nc
        return df


    def get_freq_features(df, frequency_map):
        df["freq1"] = df["q1"].map(lambda x: frequency_map[x])
        df["freq2"] = df["q2"].map(lambda x: frequency_map[x])
        return df


    train_df = train[['qid1', 'qid2']]
    test_df = test[['qid1', 'qid2']]


    print("Hashing the questions...")
    question_dict = create_question_hash(train_df, test_df)
    train_df = get_hash(train_df, question_dict)
    test_df = get_hash(test_df, question_dict)
    print("Number of unique questions:", len(question_dict))

    print("Calculating kcore features...")
    all_df = pd.concat([train_df, test_df])
    kcore_dict = get_kcore_dict(all_df)
    train_df = get_kcore_features(train_df, kcore_dict)
    test_df = get_kcore_features(test_df, kcore_dict)
    train_df = convert_to_minmax(train_df, "kcore")
    test_df = convert_to_minmax(test_df, "kcore")

    print("Calculating common neighbor features...")
    neighbors = get_neighbors(train_df, test_df)
    train_df = get_neighbor_features(train_df, neighbors)
    test_df = get_neighbor_features(test_df, neighbors)


    print("Calculating frequency features...")
    frequency_map = dict(zip(*np.unique(np.vstack((all_df["q1"], all_df["q2"])), return_counts=True)))
    train_df = get_freq_features(train_df, frequency_map)
    test_df = get_freq_features(test_df, frequency_map)
    train_df = convert_to_minmax(train_df, "freq")
    test_df = convert_to_minmax(test_df, "freq")


    cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq", 'q1',
            'q2']
    features_magic_train = train_df.loc[:, cols]
    features_magic_test = test_df.loc[:, cols]


    path = '../cache/'

    G = nx.Graph()
    for q1, q2 in train[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    for q1, q2 in test[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    A = nx.adjacency_matrix(G)
    nmf = NMF(n_components=2, random_state=2018)
    adjacency_matrix_nmf = nmf.fit_transform(A)

    A = nx.incidence_matrix(G)
    nmf = NMF(n_components=2, random_state=2018)
    incidence_matrix_nmf = nmf.fit_transform(A)

    nodes = G.nodes()

    d = dict()
    for n, a, i in zip(nodes, adjacency_matrix_nmf, incidence_matrix_nmf):
        d[n] = np.concatenate([a, i])

    train_q1_decom = np.vstack(train['qid1'].apply(lambda x: d[x]).values.tolist())
    train_q2_decom = np.vstack(train['qid2'].apply(lambda x: d[x]).values.tolist())
    test_q1_decom = np.vstack(test['qid1'].apply(lambda x: d[x]).values.tolist())
    test_q2_decom = np.vstack(test['qid2'].apply(lambda x: d[x]).values.tolist())

    train_decom_diff = pd.DataFrame(np.abs(train_q1_decom - train_q2_decom),
                                    columns=["decom_diff_%s" % i for i in range(4)])
    train_decom_angle = pd.DataFrame(train_q1_decom * train_q2_decom, columns=["decom_angle_%s" % i for i in range(4)])

    test_decom_diff = pd.DataFrame(np.abs(test_q1_decom - test_q2_decom),
                                   columns=["decom_diff_%s" % i for i in range(4)])
    test_decom_angle = pd.DataFrame(test_q1_decom * test_q2_decom, columns=["decom_angle_%s" % i for i in range(4)])


    G = nx.Graph()
    for q1, q2 in train[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    for q1, q2 in test[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    avg_degrees = nx.average_neighbor_degree(G)

    def hash_subgraph(q1, q2):
        q1_idf = avg_degrees.get(q1, 0)
        q2_idf = avg_degrees.get(q2, 0)
        qmax = max(q1_idf, q2_idf)
        qmin = min(q1_idf, q2_idf)
        qdiff = qmax - qmin
        qmean = 0.5 * (q1_idf + q2_idf)
        return [qmax, qmin, qdiff, qmean]


    def extract_hash_subgraph_features(df):
        print("hash_subgraph features...")
        hash_subgraph_features = [hash_subgraph(x[0], x[1]) for x in df[['qid1', 'qid2']].values]
        df["hash_subgraph_qmax"] = list(map(lambda x: x[0], hash_subgraph_features))
        df["hash_subgraph_qmin"] = list(map(lambda x: x[1], hash_subgraph_features))
        df["hash_subgraph_qdiff"] = list(map(lambda x: x[2], hash_subgraph_features))
        df["hash_subgraph_qmean"] = list(map(lambda x: x[3], hash_subgraph_features))
        del hash_subgraph_features
        gc.collect()


    extract_hash_subgraph_features(train)
    extract_hash_subgraph_features(test)

    features_hash_subgraph_train = train[
        ["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]
    features_hash_subgraph_test = test[
        ["hash_subgraph_qmax", "hash_subgraph_qmin", "hash_subgraph_qdiff", "hash_subgraph_qmean"]]


    G = nx.Graph()
    for q1, q2 in train[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    for q1, q2 in test[['qid1', 'qid2']].values:
        G.add_edge(q1, q2)

    pr = nx.pagerank(G, alpha=0.9)


    def pagerank(q1, q2):
        pr1 = pr[q1] * 1e6
        pr2 = pr[q2] * 1e6
        return [max(pr1, pr2), min(pr1, pr2), (pr1 + pr2) / 2.]


    def extract_pagerank_features(df):
        print("pagerank features...")
        pagerank_features = [pagerank(x[0], x[1]) for x in df[['qid1', 'qid2']].values]
        df["max_pr"] = list(map(lambda x: x[0], pagerank_features))
        df["min_pr"] = list(map(lambda x: x[1], pagerank_features))
        df["mean_pr"] = list(map(lambda x: x[2], pagerank_features))
        del pagerank_features
        gc.collect()


    extract_pagerank_features(train)
    extract_pagerank_features(test)

    pagerank_features_train = train[["max_pr", "min_pr", "mean_pr"]]
    pagerank_features_test = test[["max_pr", "min_pr", "mean_pr"]]

    features_train = pd.concat([features_magic_train, train_decom_angle, train_decom_diff, features_hash_subgraph_train,
                                pagerank_features_train], axis=1).values
    features_test = pd.concat(
        [features_magic_test, test_decom_angle, test_decom_diff, features_hash_subgraph_test, pagerank_features_test],
        axis=1).values


    dump((features_train, features_test), "./cache/features_aug.jl")
else:
    features_train,features_test =load("./cache/features_aug.jl")
