from argparse import ArgumentParser
import os
import numpy as np

parser = ArgumentParser()
parser.add_argument("log_dir", type=str)
parser.add_argument("out_fname", type=str)
parser.add_argument("num_seeds", type=int)

datasets = ["higgs", "epsilon", "kitsune", "criteo", "bosch", "year", "yahoo", "msltr"]

def parse(log_dir, out_fname, num_seeds):
    results = {}
    for fname in os.listdir(log_dir):
        _, data, seed, bins = fname.split(".")[0].split("_")
        if data not in results:
            results[data] = {}
        if bins not in results[data]:
            results[data][bins] = np.zeros(num_seeds)
        seed = int(seed.split("seed")[-1])
        compare_func = max if data != "year" else min
        metric_val = 0.0 if compare_func == max else 1000000.0
        val_index = "valid_1 " if (data != "yahoo" and data != "msltr") else "valid_1 ndcg@10 "
        with open(f"{log_dir}/{fname}", "r") as in_file:
            for line in in_file:
                if line.find(val_index) != -1:
                    metric_val = compare_func(metric_val, float(line.strip().split(" ")[-1]))
        results[data][bins][seed] = metric_val

    with open(out_fname, "w") as out_file:
        out_file.write("|")
        binss = np.sort(list(results[datasets[0]].keys()))
        for data in datasets:
            out_file.write(f" {data} |")
        out_file.write("\n")
        out_file.write("|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for bins in binss:
            out_file.write("| ")
            for data in datasets:
                mean = np.mean(results[data][bins])
                std = np.std(results[data][bins])
                out_file.write(f" {mean:.6f}/{std:.6f} |")
            out_file.write("\n")

if __name__ == "__main__":
    args = parser.parse_args()
    parse(args.log_dir, args.out_fname, args.num_seeds)
