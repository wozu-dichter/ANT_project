import os
import numpy as np
import matplotlib.pyplot as plt
import re
from custom_lib import load_npy, glob_sorted, plt_show_full_screen
from scipy.signal import welch
from band_pass_filter import butter_bandpass_filter


def get_file_paths(file_dir, data_type, suffix):
    paths = [p for p in glob_sorted(file_dir + "/*{}".format(suffix)) if data_type in p]
    return paths


def apply_normalization(signal, mode):
    available_modes = ["min_max", "z_score", "mean_norm"]
    assert mode in available_modes

    if mode == "min_max":
        signal = signal - np.min(signal)
        signal = signal / np.max(signal)
    elif mode == "z_score":
        signal = signal - np.mean(signal)
        signal = signal / np.std(signal)
    elif mode == "mean_norm":
        signal = signal - np.mean(signal)

    return signal


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def get_fatigue_level(file_dir):
    with open(file_dir + "/fatigue_level.txt", "r") as f:
        lines = f.readlines()

    high = re.split('[:\n]', lines[0])[1]
    low = re.split("[:\n]", lines[1])[1]

    return high, low





dataset_dir = "./dataset"
channel_orders = ["Fp1", "Fp2", "F3", "Fz", "F4",
                  "T7", "C3", "Cz", "C4", "T8",
                  "P3", "Pz", "P4", "P7", "P8",
                  "Oz"]

selected_channels = ["P3", "Pz", "P4", "P7", "P8", "Oz"]
colors = ["r", "g", "b", "k"]

num_channels = len(channel_orders)
sample_rate = 500

plt.ion()

subject_dirs = glob_sorted(dataset_dir + "/*")
for nth_subject, subject_dir in enumerate(subject_dirs):
    record_dirs = glob_sorted(subject_dir + "/*")
    subject_id = os.path.basename(subject_dir)
    for nth_record, record_dir in enumerate(record_dirs):
        npy_paths = get_file_paths(file_dir=record_dir, data_type="rest", suffix=".npy")
        highest, lowest = get_fatigue_level(file_dir=record_dir)
        for nth_channel, selected_channel in enumerate(selected_channels):
            print("subject : {} / {}, record : {} / {}, channels : {} / {}"
                  .format(nth_subject + 1, len(subject_dirs),
                          nth_record + 1, len(record_dirs),
                          nth_channel + 1, len(selected_channels)))
            channel_dir = "psd_figures2/{}".format(selected_channel)
            create_dir(channel_dir)
            plt.clf()
            plt_show_full_screen()
            for nth_stage, (npy_path, color) in enumerate(zip(npy_paths, colors)):
                npy_data = load_npy(npy_path)
                raw_eeg = npy_data["eeg"]
                channel_index = channel_orders.index(selected_channel)
                channel_data = raw_eeg[..., channel_index].reshape(-1)
                channel_data = butter_bandpass_filter(channel_data, lowcut=0.1, highcut=50, fs=sample_rate)
                channel_data = apply_normalization(channel_data, mode="mean_norm")
                f, Pxx_den = welch(channel_data, sample_rate, nperseg=sample_rate)
                plt.semilogy(f, Pxx_den, color=color, label="rest{}".format(nth_stage + 1))
            plt.legend(loc="upper right")
            plt.xlim(left=0, right=35)
            plt.ylim(bottom=1e-2)
            plt.axvline(x=4, linestyle="--", color="k")
            plt.axvline(x=8, linestyle="--", color="k")
            plt.axvline(x=13, linestyle="--", color="k")
            plt.xlabel("frequency [Hz] highest : {}, lowest : {}".format(highest, lowest))
            plt.ylabel("PSD [V**2/Hz]")
            plt.pause(0.01)

            plt.savefig("{}/psd_{}_{}_{}.png".format(channel_dir, subject_id, nth_record + 1, selected_channel))
