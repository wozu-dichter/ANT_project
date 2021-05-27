import os
import matplotlib.pyplot as plt
from custom_lib import load_npy, glob_sorted, plt_show_full_screen


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def get_file_paths(file_dir, data_type, suffix):
    paths = [p for p in glob_sorted(file_dir + "/*{}".format(suffix)) if data_type in p]
    return paths


dataset_dir = "./dataset"
gt_types = ["", "Congruent", "Incongruent", "No Target", ""]
response_times_all = {"Congruent": [], "Incongruent": [], "noTarget": []}

num_channels = 16
sample_rate = 500
start_time = 2
end_time = 3

plt.ion()

for subject_dir in glob_sorted(dataset_dir + "/*"):
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = get_file_paths(file_dir=record_dir, data_type="session", suffix=".npy")
        for nth_stage, npy_path in enumerate(npy_paths):
            npy_data = load_npy(npy_path)
            raw_eeg = npy_data["eeg"]
            multi_trial = npy_data["multiTrialData"]

            response_times = {"Congruent": [], "Incongruent": [], "noTarget": []}

            for nth_trial, single_trial in multi_trial.items():
                if not (single_trial["hasAnswer"] and single_trial["match"]):
                    continue
                trial_gt = single_trial["groundTruth"]
                response_times[trial_gt].append(single_trial["responseTime"])
                response_times_all[trial_gt].append(single_trial["responseTime"])

            data = []
            plt.figure(1)
            plt.clf()

            for nth_type, (gt, rts) in enumerate(response_times.items()):
                data.append(rts)
                plt.subplot(3, 1, nth_type + 1)
                plt.xlabel(gt)
                plt.hist(rts, bins=20)
            plt.subplots_adjust(hspace=0.5)
            plt.savefig("{}/rt_histograms_{}.png".format(record_dir, nth_stage + 1))
            plt.pause(0.01)

            plt.figure(2)
            plt.clf()

            plt.boxplot(data, showfliers=False, showmeans=True)
            plt.xticks(list(range(len(gt_types))), gt_types)
            plt.xlabel("Conditions")
            plt.ylabel("Response times")
            plt.pause(0.01)
            plt.savefig("{}/rt_boxplot_{}.png".format(record_dir, nth_stage + 1))

data = []
plt.figure(1)
plt.clf()

for nth_type, (gt, rts) in enumerate(response_times_all.items()):
    data.append(rts)
    plt.subplot(3, 1, nth_type + 1)
    plt.xlabel(gt)
    plt.hist(rts, bins=20)
plt.subplots_adjust(hspace=0.5)
plt.savefig("rt_histograms_all.png")
plt.pause(0.01)

plt.figure(2)
plt.clf()

plt.boxplot(data, showfliers=False, showmeans=True)
plt.xticks(list(range(len(gt_types))), gt_types)
plt.xlabel("Conditions")
plt.ylabel("Response times")
plt.pause(0.01)
plt.savefig("rt_boxplot_all.png")
