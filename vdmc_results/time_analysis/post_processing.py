import os
import pickle
import itertools
import numpy as np


def validate_same_results():
    from generate_graphs import SIZES, AVG_DEG, DIRECTED, NUM_RUNS
    levels = [3, 4]
    for size, degree, is_directed, run, level in itertools.product(SIZES, AVG_DEG, DIRECTED, range(NUM_RUNS), levels):
        dirname = "size%s_deg%d_directed%r_runs" % (size, degree, is_directed)
        dir_path = os.path.join(dirname, "run_%d" % run)
        with open(os.path.join(dir_path, "motifs_python", "motif%d.pkl" % level), "rb") as py:
            res_py = pickle.load(py)
        with open(os.path.join(dir_path, "motifs_cpp", "motif%d.pkl" % level), "rb") as cpp:
            res_cpp = pickle.load(cpp)
        with open(os.path.join(dir_path, "motifs_gpu", "motif%d.pkl" % level), "rb") as gpu:
            res_gpu = pickle.load(gpu)

        ftr_py, ftr_cpp, ftr_gpu = map(res_to_matrix, [res_py, res_cpp, res_gpu])
        ftr_py, ftr_cpp = map(res_to_matrix, [res_py, res_cpp])
        if not np.array_equal(ftr_py, ftr_cpp):
            print("size %s, deg %d, directed %r, run %d, motif %d, Python != C++" % (size, degree, is_directed, run, level))
        if not np.array_equal(ftr_py, ftr_gpu):
            print("size %s, deg %d, directed %r, run %d, motif %d, Python != GPU" % (size, degree, is_directed, run, level))
        if not np.array_equal(ftr_cpp, ftr_gpu):
            print("size %s, deg %d, directed %r, run %d, motif %d, C++ != GPU" % (size, degree, is_directed, run, level))


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


def aggregate_results():
    # Assuming validation is complete.
    from additional_features import MotifProbability
    from generate_graphs import SIZES, AVG_DEG, DIRECTED, NUM_RUNS
    if not os.path.exists("results"):
        os.mkdir("results")
    for size, degree, is_directed in itertools.product(SIZES, AVG_DEG, DIRECTED):
        p = min([1, float(degree) / (size - 1) * (0.5 if is_directed else 1)])
        mp = MotifProbability(size=size, edge_probability=p, clique_size=0, directed=is_directed)
        expected_motifs = {"motif3": [mp.motif_expected_non_clique_vertex(motif_index=idx)
                                      for idx in get_motif_indices(3, is_directed)],
                           "motif4": [mp.motif_expected_non_clique_vertex(motif_index=idx)
                                      for idx in get_motif_indices(4, is_directed)]}
        dirname = "size%s_deg%d_directed%r_runs" % (size, degree, is_directed)
        if not os.path.exists(os.path.join("results", dirname)):
            os.mkdir(os.path.join("results", dirname))
            for run in range(NUM_RUNS):
                os.mkdir(os.path.join("results", dirname, "run_%d" % run))
        with open(os.path.join("results", dirname, "expected_motifs.pkl"), "wb") as e_motifs_file:
            pickle.dump(expected_motifs, e_motifs_file)
        times_elapsed = {"motif3_python": [], "motif4_python": [],
                         "motif3_cpp": [], "motif4_cpp": [],
                         "motif3_gpu": [], "motif4_gpu": []}
        for run in range(NUM_RUNS):
            dir_path = os.path.join(dirname, "run_%d" % run)
            res_3 = os.path.join(dir_path, "motifs_gpu", "motif3.pkl")
            res_4 = os.path.join(dir_path, "motifs_gpu", "motif4.pkl")
            dump_3 = os.path.join("results", dir_path, "motif3.pkl")
            dump_4 = os.path.join("results", dir_path, "motif4.pkl")
            results_to_folder(res_3, dump_3)
            results_to_folder(res_4, dump_4)
            for level, mode in itertools.product([3, 4], ["python", "cpp", "gpu"]):
                if all([size == 10000, degree == 100, level == 4, mode == "python"]):
                    continue
                path_to_time_log = os.path.join(dir_path, "motifs_%s" % mode, "CalculationLogger%d.log" % level)
                t = get_calc_times(path_to_time_log)
                times_elapsed["motif%d_%s" % (level, mode)].append(t)
        with open(os.path.join("results", dirname, "times_elapsed.pkl"), "wb") as times_file:
            pickle.dump(times_elapsed, times_file)


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


def results_to_folder(res_path, dump_path):
    with open(res_path, "rb") as result_file:
        with open(dump_path, "wb") as dump_file:
            res = pickle.load(result_file)
            ftr = res_to_matrix(res)
            pickle.dump(ftr, dump_file)


def get_calc_times(time_path):  # IN SECONDS
    time_taken = 0
    with open(time_path, "r") as log_file:
        for i, line in enumerate(log_file):
            if i == 1:
                time_str = line.strip().split(" ")[-1].split(":")
                time_taken = 3600 * int(time_str[0]) + 60 * int(time_str[1]) + float(time_str[2])
    return time_taken


if __name__ == "__main__":
    validate_same_results()
    # aggregate_results()
