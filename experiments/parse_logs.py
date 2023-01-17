from argparse import ArgumentParser
import os
import numpy as np

parser = ArgumentParser()
parser.add_argument("log_dir", type=str)
parser.add_argument("out_fname", type=str)
parser.add_argument("num_seeds", type=int)
parser.add_argument("algorithm", type=str)
parser.add_argument("--for-speed", action='store_true')

datasets = ["higgs", "epsilon", "kitsune", "criteo", "bosch", "year", "yahoo", "msltr"]

def parse(log_dir, out_fname, num_seeds, algorithm, for_speed):
    results = {}
    for fname in os.listdir(log_dir):
        _, data, seed, bins = fname.split(".")[0].split("_")
        if data not in results:
            results[data] = {}
        if bins not in results[data]:
            results[data][bins] = np.zeros(num_seeds)
            if algorithm == "lgb" and bins != 'xgb' and bins != 'cat' and for_speed:
                results[data][f"{bins} hist"] = np.zeros(num_seeds)
        seed = int(seed.split("seed")[-1])
        compare_func = max if data != "year" else min
        metric_val = 0.0 if compare_func == max else 1000000.0
        time_val = 0.0
        if not for_speed:
            if (algorithm == "lgb" or fname.find("fp32") != -1 or fname.find("bins") != -1) and (fname.find("cat") == -1 and fname.find("xgb") == -1):
                val_index = "valid_1 " if (data != "yahoo" and data != "msltr") else "valid_1 ndcg@10 "
                with open(f"{log_dir}/{fname}", "r") as in_file:
                    for line in in_file:
                        if line.find(val_index) != -1:
                            metric_val = compare_func(metric_val, float(line.strip().split(" ")[-1]))
            elif algorithm == "cat" or fname.find("cat") != -1:
                with open(f"{log_dir}/{fname}", "r") as in_file:
                    for line in in_file:
                        if line.find("test:") != -1:
                            metric_val = compare_func(metric_val, float(line.strip().split("test:")[1].strip().split(" ")[0].split("best:")[0].strip()))
            elif algorithm == "xgb" or fname.find("xgb") != -1:
                with open(f"{log_dir}/{fname}", "r") as in_file:
                    for line in in_file:
                        if line.find("test-") != -1:
                            metric_val = compare_func(metric_val, float(line.strip().split(":")[-1]))
            results[data][bins][seed] = metric_val
        else:
            if (algorithm == "lgb" or fname.find("fp32") != -1 or fname.find("bins") != -1) and (fname.find("cat") == -1 and fname.find("xgb") == -1):
                with open(f"{log_dir}/{fname}", "r") as in_file:
                    for line in in_file:
                        if line.find("seconds elapsed") != -1:
                            time_val = float(line.strip().split("[Info]")[-1].strip().split(" ")[0])
                        if line.find("ConstructHistograms costs:") != -1 or line.find("ConstructHistogramForLeaf costs:") != -1:
                            hist_time_val = float(line.strip().split(" ")[-1])
                results[data][f"{bins} hist"][seed] = hist_time_val
            elif algorithm == "xgb" or fname.find("xgb") != -1:
                    all_times = []
                    with open(f"{log_dir}/{fname}", "r") as in_file:
                        for line in in_file:
                            if line.find("sec elapsed") != -1:
                                time_val = float(line.strip().split("sec")[0].strip().split(" ")[-1])
                                all_times.append(time_val)
                    time_val = all_times[-1] - all_times[1] # subtract time of 1st iteration which contains some time for data preprocessing
            elif algorithm == "cat" or fname.find("cat") != -1:
                with open(f"{log_dir}/{fname}", "r") as in_file:
                    for line in in_file:
                        if line.find("total:") != -1:
                            time_str = line.strip().split("total:")[-1].strip().split("remaining:")[0].strip()
                            time_str_splits = time_str.split(" ")
                            time_val = 0.0
                            for time_str_split in time_str_splits:
                                if time_str_split.endswith("ms"):
                                    time_val += float(time_str_split[:-2]) * 1e-3
                                elif time_str_split.endswith("s"):
                                    time_val += float(time_str_split[:-1])
                                elif time_str_split.endswith("m"):
                                    time_val += float(time_str_split[:-1]) * 60.0 
                                elif time_str_split.endswith("h"):
                                    time_val += float(time_str_split[:-1]) * 3600.0
            results[data][bins][seed] = time_val

    with open(out_fname, "w") as out_file:
        out_file.write(f"| algorithm |")
        binss = np.hstack([np.sort(list(filter(lambda x: x.find("hist") == -1, results[datasets[0]].keys()))),
                np.sort(list(filter(lambda x: x.find("hist") != -1, results[datasets[0]].keys())))])
        for data in datasets:
            out_file.write(f" {data} |")
        out_file.write("\n")
        out_file.write(f"|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for bins in binss:
            out_file.write(f"| {bins} |")
            for data in datasets:
                mean = np.mean(results[data][bins])
                std = np.std(results[data][bins])
                out_file.write(f" {mean:.6f}/{std:.6f} |")
            out_file.write("\n")
    with open(out_fname + ".mean.tex", "w") as out_file:
        out_file.write(f"| algorithm |")
        binss = np.hstack([np.sort(list(filter(lambda x: x.find("hist") == -1, results[datasets[0]].keys()))),
                np.sort(list(filter(lambda x: x.find("hist") != -1, results[datasets[0]].keys())))])
        for data in datasets:
            out_file.write(f" {data} |")
        out_file.write("\n")
        out_file.write(f"|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for bins in binss:
            out_file.write(f"& {bins} &")
            for data in datasets:
                mean = np.mean(results[data][bins])
                out_file.write(f" {mean:.6f} &")
            out_file.write("\n")
    with open(out_fname + ".std.tex", "w") as out_file:
        out_file.write(f"| algorithm |")
        binss = np.hstack([np.sort(list(filter(lambda x: x.find("hist") == -1, results[datasets[0]].keys()))),
                np.sort(list(filter(lambda x: x.find("hist") != -1, results[datasets[0]].keys())))])
        for data in datasets:
            out_file.write(f" {data} |")
        out_file.write("\n")
        out_file.write(f"|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for bins in binss:
            out_file.write(f"& {bins} &")
            for data in datasets:
                std = np.std(results[data][bins])
                out_file.write(f" {std:.6f} &")
            out_file.write("\n")
    with open(out_fname + ".tex", "w") as out_file:
        out_file.write(f"| algorithm |")
        binss = np.hstack([np.sort(list(filter(lambda x: x.find("hist") == -1, results[datasets[0]].keys()))),
                np.sort(list(filter(lambda x: x.find("hist") != -1, results[datasets[0]].keys())))])
        for data in datasets:
            out_file.write(f" {data} |")
        out_file.write("\n")
        out_file.write(f"|-------|-------|-------|-------|-------|-------|-------|-------|-------|\n")
        for bins in binss:
            out_file.write(f"& {bins} &")
            for data in datasets:
                mean = np.mean(results[data][bins])
                out_file.write(f" {mean:.6f} &")
            out_file.write("\n")
            out_file.write(f"& {bins} &")
            for data in datasets:
                std = np.std(results[data][bins])
                std_str = f"{std:.6f}".lstrip('0')
                out_file.write(f" $\pm{std_str}$ &")
            out_file.write("\n")



if __name__ == "__main__":
    args = parser.parse_args()
    parse(args.log_dir, args.out_fname, args.num_seeds, args.algorithm, args.for_speed)
