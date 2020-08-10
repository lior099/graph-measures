import os
import pickle
import itertools
import numpy as np


def aggregate_results():
    # Assuming validation is complete.
    from additional_features import MotifProbability
    from generate_graphs_2 import SIZES, PROB, DIRECTED, NUM_RUNS
    if not os.path.exists("results_2"):
        os.mkdir("results_2")
    expected_motifs = {}
    for size, is_directed in itertools.product(SIZES, DIRECTED):
        mp = MotifProbability(size=size, edge_probability=PROB, clique_size=0, directed=is_directed)
        expected_motifs.update({"n_{}_{}_motif3".format(size, "d" if is_directed else "ud"):
                               [mp.motif_expected_non_clique_vertex(motif_index=idx) for idx in get_motif_indices(3, is_directed)],
                                "n_{}_{}_motif4".format(size, "d" if is_directed else "ud"):
                                    [mp.motif_expected_non_clique_vertex(motif_index=idx) for idx in get_motif_indices(4, is_directed)]})
        dirname = "size{}_p{}_directed{}_runs".format(size, PROB, is_directed)
        res = {"motif3": [], "motif4": []}
        for run in range(NUM_RUNS):
            dir_path = os.path.join(dirname, "run_" + str(run))
            for level in [3, 4]:
                res["motif" + str(level)].append(
                    res_to_matrix(pickle.load(open(os.path.join(dir_path, "motifs_gpu", "motif{}.pkl".format(level)), "rb"))))
        for level in [3, 4]:
            res_lvl = np.vstack(res["motif" + str(level)])
            np.savetxt(os.path.join("results_2", "size{}_p{}_directed{}_motif{}.csv".format(size, PROB, is_directed, level)),
                       res_lvl, delimiter=",")
    pickle.dump(expected_motifs, open(os.path.join("results_2", "expected_motifs.pkl"), 'wb'))


def res_to_matrix(res):
    ftr = res._features
    if type(ftr) == list:
        return np.array(ftr, dtype=np.int64)
    else:  # Dict
        mx = np.zeros((len(ftr), len(ftr[0]) - 1), dtype=np.int64)
        for i in range(len(ftr)):
            for j in range(len(ftr[0]) - 1):
                mx[i, j] = ftr[i][j]
        return mx


def get_motif_indices(level, directed):
    if directed:
        if level == 3:
            return range(13)
        else:
            return range(13, 212)
    else:
        if level == 3:
            return range(2)
        else:
            return range(2, 8)


if __name__ == "__main__":
    aggregate_results()
