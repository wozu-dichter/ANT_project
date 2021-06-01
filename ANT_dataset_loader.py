import os
import re
import warnings
import random
import platform
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from termcolor import cprint
from tqdm import tqdm
from scipy.signal import stft, butter, sosfilt, filtfilt


class DatasetLoader:
    def __init__(self):
        # fixed parameters
        # NO NEED TO MODIFY
        self.figures_dir = "./figures"
        self.__init_settings()

        # parameters initialization
        # NO NEED TO MODIFY
        self.data_type = None
        self.feature_type = None
        self.subjects_npy_paths = None
        self.subjects_fatigue_levels = None
        self.subjects_trials_data = None
        self.reformatted_data = None
        self.selected_channels = None

        # # optional and preprocessing parameters
        self.dataset_dir = "./dataset2"
        self.num_channels = 32
        self.sample_rate = 512  # sample point
        self.channel_orders = {"Fp1": 0, "Fp2": 1, "F3": 2, "Fz": 3, "F4": 4, "T7": 5, "C3": 6, "Cz": 7,
                               "C4": 8, "T8": 9, "P3": 10, "Pz": 11, "P4": 12, "P7": 13, "P8": 14, "Oz": 15,
                               "AF3": 16, "AF4": 17, "F7": 18, "F8": 19, "FT7": 20, "FC3": 21, "FCz": 22, "FC4": 23,
                               "FT8": 24, "TP7": 25, "CP3": 26, "CPz": 27, "CP4": 28, "TP8": 29, "O1": 30, "O2": 31}

        # # optional and preprocessing parameters
        # save RT boxplot or not
        self.save_box_plot = False

        # parameters for data reformat
        self.shuffle = True
        self.validation_split = 0.1

        # signal length in one rest trial, unit in seconds
        self.rest_signal_len = 5

        # signal interval selection in one session trial, unit in seconds
        self.session_signal_start = 2
        self.session_signal_end = 4

        # parameters for normalization method
        self.apply_signal_normalization = True
        self.normalization_mode = "min_max"
        # parameters for RT outlier filtering, unit in percentage
        self.apply_rt_thresholding = True
        self.rt_filtration_rate = 0.05

        # parameters for Butterworth bandpass filter
        self.apply_bandpass_filter = True
        self.bandpass_visualize = False
        self.bandpass_filter_type = "filtfilt"
        self.bandpass_filter_order = 1
        self.bandpass_low_cut = 5
        self.bandpass_high_cut = 28

        # parameters for STFT
        self.stft_visualize = False
        self.stft_nperseg = 500
        self.stft_noverlap_ratio = 0.95
        self.stft_min_freq = 3
        self.stft_max_freq = 28

    def __init_settings(self):
        warnings.filterwarnings("error")
        plt.ion()
        create_dir(self.figures_dir)

    def load_data(self, data_type, feature_type, fatigue_basis="by_feedback", single_subject="all", excluded_subjects=None, selected_channels=None):
        self.__parameter_check(data_type, feature_type, fatigue_basis, single_subject, excluded_subjects, selected_channels)

        self.data_type = data_type
        self.feature_type = feature_type
        self.selected_channels = selected_channels

        self.__get_npy_file_paths(excluded_subjects, single_subject, fatigue_basis)
        self.__load_trials_data()
        self.__trials_data_preprocessing()
        self.__data_reformat()

        return self.subjects_trials_data, self.reformatted_data

    def __parameter_check(self, data_type, feature_type, fatigue_basis, single_subject, excluded_subjects, selected_channels):
        # parameters in self.__init__:----------------------------------------------------------------------------------------------------------------
        # check normalization_mode
        available_normalization_modes = ["min_max", "z_score", "mean_norm"]
        assert self.normalization_mode in available_normalization_modes, "normalization_mode must be one of {}".format(available_normalization_modes)

        # parameters in self.load_data:---------------------------------------------------------------------------------------------------------------
        # check data_type
        available_data_types = ["session", "rest"]
        assert data_type in available_data_types, "data_type must be one of {}".format(available_data_types)

        # check feature_type
        available_feature_types = ["time", "stft", "wavelet"]
        assert feature_type in available_feature_types, "feature_type must be one of {}".format(available_feature_types)

        # check fatigue_basis
        available_fatigue_basis = ["by_feedback", "by_time"]
        assert fatigue_basis in available_fatigue_basis, "fatigue_basis must be one of {}".format(available_fatigue_basis)

        # check single_subject
        available_subjects = self.get_subject_ids() + ["all"]
        assert single_subject in available_subjects, "single_subject must be one of {}".format(available_subjects)

        # check excluded_subjects
        if excluded_subjects:
            for subject in excluded_subjects:
                assert subject in self.get_subject_ids(), "unknown subject : {}".format(subject)

        # check selected_channels
        if selected_channels:
            for channel in selected_channels:
                assert channel in self.channel_orders.keys(), "unknown channel : {}".format(channel)

        # warning
        if single_subject != "all" and bool(excluded_subjects):
            print_warning("excluded_subjects and single_subject are given at the same time, excluded_subjects may be unnecessary.")

    def __get_npy_file_paths(self, excluded_subjects, single_subject, fatigue_basis):
        self.subjects_npy_paths = {}
        self.subjects_fatigue_levels = {}

        if single_subject == "all":
            subject_dirs = glob_sorted(self.dataset_dir + "/*")
        else:
            subject_dirs = ["{}/{}".format(self.dataset_dir, single_subject)]

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            if bool(excluded_subjects) and subject_id in excluded_subjects:
                print_info("Skip subject, {}".format(subject_id))
                continue
            subject_npy_paths = []
            subject_fatigue_levels = {"high": [], "low": []}
            for record_dir in glob_sorted(subject_dir + "/*"):
                npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if self.data_type in p]
                subject_npy_paths.extend(npy_paths)
                high, low = get_fatigue_level(record_dir, fatigue_basis)
                subject_fatigue_levels["high"].append(high)
                subject_fatigue_levels["low"].append(low)
            self.subjects_npy_paths[subject_id] = custom_sort(subject_npy_paths)
            self.subjects_fatigue_levels[subject_id] = subject_fatigue_levels

    def __load_trials_data(self):
        if self.selected_channels:
            channel_ids = [self.channel_orders[channel_name] for channel_name in self.selected_channels]
        else:
            self.selected_channels = list(self.channel_orders.keys())[:self.num_channels]
            channel_ids = list(self.channel_orders.values())[:self.num_channels]
        desired_channel_num = len(channel_ids)

        self.subjects_trials_data = {}
        for nth_subject, (subject_id, npy_paths) in enumerate(self.subjects_npy_paths.items()):
            self.subjects_trials_data[subject_id] = []
            progressbar_prefix = "Loading npy files, {}, {}/{}".format(subject_id, nth_subject + 1, len(self.subjects_npy_paths))
            for nth_npy, npy_path in enumerate(tqdm_info(npy_paths, prefix=progressbar_prefix)):
                npy_data = load_npy(npy_path)
                raw_eeg = npy_data["eeg"][:, :, channel_ids]

                if self.data_type == "session":
                    multi_trial_data = npy_data["multiTrialData"]
                    for nth_trial, (key, trial_data) in enumerate(multi_trial_data.items()):
                        if not (trial_data["hasAnswer"] and trial_data["match"]):
                            continue
                        trial_data["eeg"] = raw_eeg[nth_trial * 4:(nth_trial + 1) * 4].reshape(-1, desired_channel_num)
                        self.subjects_trials_data[subject_id].append(trial_data)

                elif self.data_type == "rest":
                    for i in range(0, raw_eeg.shape[0], self.rest_signal_len):
                        trial_data = {"eeg": raw_eeg[i:i + self.rest_signal_len].reshape(-1, desired_channel_num)}
                        if npy_path in self.subjects_fatigue_levels[subject_id]["high"]:
                            trial_data["fatigue_level"] = "high"
                        elif npy_path in self.subjects_fatigue_levels[subject_id]["low"]:
                            trial_data["fatigue_level"] = "low"
                        else:
                            trial_data["fatigue_level"] = None
                        self.subjects_trials_data[subject_id].append(trial_data)

    def __trials_data_preprocessing(self):
        tic = time.time()
        for nth_subject, (subject_id, trials_data) in enumerate(self.subjects_trials_data.items()):
            if self.apply_signal_normalization:
                self.__signal_normalization(subject_id)
            if self.data_type == "session" and self.apply_rt_thresholding:
                self.__rt_thresholding(subject_id, save_box_plot=self.save_box_plot)
            if self.data_type == "session":
                self.__rt_normalization(subject_id)
            if self.apply_bandpass_filter:
                self.__bandpass_filter(subject_id)

        # self.__rt_distribution_balance()
        self.__time_frequency_analysis()

        print_info("Total time spent on data preprocessing: {:.2f} seconds".format(time.time() - tic))

    def __rt_distribution_balance(self, rt_distribution_visualize=True):

        rts_before = {"Congruent": [], "Incongruent": [], "noTarget": []}
        rts_after = {"Congruent": [], "Incongruent": [], "noTarget": []}

        for nth_subject, (subject_id, trials_data) in enumerate(self.subjects_trials_data.items()):
            for trial_data in trials_data:
                rts_before[trial_data["groundTruth"]].append(trial_data["normalized_rt"])
                trial_data["normalized_rt"] = sigmoid(trial_data["normalized_rt"], x_scale=5, x_shift=0.3)
                rts_after[trial_data["groundTruth"]].append(trial_data["normalized_rt"])

        if rt_distribution_visualize:
            plt.figure("before balance")
            plt.clf()
            plt.subplots_adjust(hspace=0.5)
            for nth_type, (gt, rts) in enumerate(rts_before.items()):
                plt.subplot(3, 1, nth_type + 1)
                plt.xlabel(gt)
                plt.hist(rts, bins=20)

            plt.figure("after balance")
            plt.clf()
            plt.subplots_adjust(hspace=0.5)
            for nth_type, (gt, rts) in enumerate(rts_after.items()):
                plt.subplot(3, 1, nth_type + 1)
                plt.xlabel(gt)
                plt.hist(rts, bins=20)

            plt.waitforbuttonpress()

    def __data_reformat(self):
        self.reformatted_data = {"train_x": [], "train_y": [], "valid_x": [], "valid_y": []}
        for subject_id, trials_data in self.subjects_trials_data.items():
            print_info("Data reformat, {}".format(subject_id))

            x = self.__get_processed_x(trials_data=trials_data)
            y = self.__get_processed_y(trials_data=trials_data)
            num_trials = len(y)
            num_valid = int(round(num_trials * self.validation_split))

            if self.shuffle:
                random.seed(888)
                random_idx = random.sample(range(num_trials), num_trials)
                x = x[random_idx]
                y = y[random_idx]

            self.reformatted_data["train_x"].extend(x[num_valid:])
            self.reformatted_data["train_y"].extend(y[num_valid:])
            self.reformatted_data["valid_x"].extend(x[:num_valid])
            self.reformatted_data["valid_y"].extend(y[:num_valid])

        self.reformatted_data["train_x"] = np.array(self.reformatted_data["train_x"])
        self.reformatted_data["train_y"] = np.array(self.reformatted_data["train_y"])
        self.reformatted_data["valid_x"] = np.array(self.reformatted_data["valid_x"])
        self.reformatted_data["valid_y"] = np.array(self.reformatted_data["valid_y"])

    def __get_processed_x(self, trials_data):
        x = None
        if self.feature_type == "time":
            if self.data_type == "session":
                x = np.array([trial_data["eeg"]
                              [self.session_signal_start * self.sample_rate:
                               self.session_signal_end * self.sample_rate, :]
                              for trial_data in trials_data])
            elif self.data_type == "rest":
                x = np.array([trial_data["eeg"] for trial_data in trials_data if trial_data["fatigue_level"] in ["high", "low"]])
        elif self.feature_type == "stft":
            x = np.array([trial_data["stft_spectrum"] for trial_data in trials_data])

        return x

    def __get_processed_y(self, trials_data):
        y = None
        if self.data_type == "session":
            y = np.array([trial_data["normalized_rt"] for trial_data in trials_data])
        elif self.data_type == "rest":
            y = []
            for trial_data in trials_data:
                if trial_data["fatigue_level"] == "high":
                    y.append(1)
                elif trial_data["fatigue_level"] == "low":
                    y.append(0)
            y = np.array(y)

        return y

    def __signal_normalization(self, subject_id):

        print_info("Applying signal normalization, {}".format(subject_id))
        filtered_trials = []
        for trial_data in self.subjects_trials_data[subject_id]:
            try:
                signal = trial_data["eeg"]
                if self.normalization_mode == "min_max":
                    signal = signal - np.min(signal, axis=0)
                    signal = signal / np.max(signal, axis=0)
                elif self.normalization_mode == "z_score":
                    signal = signal - np.mean(signal, axis=0, dtype=np.float64)
                    signal = signal / np.std(signal, axis=0, dtype=np.float64)
                elif self.normalization_mode == "mean_norm":
                    signal = signal - np.mean(signal, axis=0, dtype=np.float64)
                trial_data["eeg"] = signal
                filtered_trials.append(trial_data)
            except RuntimeWarning:
                pass
        self.subjects_trials_data[subject_id] = filtered_trials

    def __rt_thresholding(self, subject_id, save_box_plot):
        if save_box_plot:
            self.__save_rt_box_plot(subject_id, state="before")

        print_info("Applying RT thresholding, {}".format(subject_id))
        trials_data = self.subjects_trials_data[subject_id]
        sorted_subject_rts = sorted([d["responseTime"] for d in trials_data])
        rt_min_threshold = sorted_subject_rts[int(len(sorted_subject_rts) * self.rt_filtration_rate)]
        rt_max_threshold = sorted_subject_rts[int(len(sorted_subject_rts) * (1 - self.rt_filtration_rate))]
        filtered_trials = [trial_data for trial_data in trials_data if rt_min_threshold <= trial_data["responseTime"] <= rt_max_threshold]
        self.subjects_trials_data[subject_id] = filtered_trials

        if save_box_plot:
            self.__save_rt_box_plot(subject_id, state="after")

    def __rt_normalization(self, subject_id):
        print_info("Applying RT normalization, {}".format(subject_id))

        rts = {"Congruent": [], "Incongruent": [], "noTarget": []}
        for trial_data in self.subjects_trials_data[subject_id]:
            gt = trial_data["groundTruth"]
            rt = trial_data["responseTime"]
            rts[gt].append(rt)

        for trial_data in self.subjects_trials_data[subject_id]:
            rt_min_value = min(rts[trial_data["groundTruth"]])
            rt_max_value = max(rts[trial_data["groundTruth"]])
            trial_data["normalized_rt"] = (trial_data["responseTime"] - rt_min_value) / (rt_max_value - rt_min_value)

    def __bandpass_filter(self, subject_id):
        progressbar_prefix = "Applying bandpass filter, {}".format(subject_id)

        if self.bandpass_filter_type == "sosfilt":
            bandpass_filter_function = butter_bandpass_filter_sosfilt
        elif self.bandpass_filter_type == "filtfilt":
            bandpass_filter_function = butter_bandpass_filter_filtfilt
        else:
            bandpass_filter_function = None

        for nth_trial, trial_data in enumerate(tqdm_info(self.subjects_trials_data[subject_id], prefix=progressbar_prefix)):
            signal_multi_channel = []
            for signal_single_channel in trial_data["eeg"].T:
                filtered_signal_copied = bandpass_filter_function(data=signal_sticking(signal_single_channel),
                                                                  low_cut=self.bandpass_low_cut,
                                                                  high_cut=self.bandpass_high_cut,
                                                                  fs=self.sample_rate,
                                                                  order=self.bandpass_filter_order)
                start = int(len(filtered_signal_copied) * (1 / 3))
                end = int(len(filtered_signal_copied) * (2 / 3))
                filtered_signal = filtered_signal_copied[start:end]
                signal_multi_channel.append(filtered_signal)

            if self.bandpass_visualize:
                num_channels = len(signal_multi_channel)
                raw_eeg_mc = trial_data["eeg"].T
                filtered_eeg_mc = np.array(signal_multi_channel)

                plt.figure("bandpass filter")
                plt.clf()
                plt_show_full_screen()
                # plt.subplots_adjust(hspace=2)

                for i, (channel_id, raw_eeg_sc, filtered_eeg_sc) in enumerate(zip(self.selected_channels, raw_eeg_mc, filtered_eeg_mc)):
                    plt.subplot(num_channels, 2, i * 2 + 1)
                    plt.plot(raw_eeg_sc)
                    plt.ylabel(channel_id, rotation=0, labelpad=20)
                    plt.yticks([])

                    plt.subplot(num_channels, 2, i * 2 + 2)
                    plt.plot(filtered_eeg_sc)
                    plt.ylabel(channel_id, rotation=0, labelpad=20)
                    plt.yticks([])

                plt.waitforbuttonpress()

            trial_data["eeg"] = np.array(signal_multi_channel).T

    def __save_rt_box_plot(self, subject_id, state):
        rts = {"Congruent": [], "Incongruent": [], "noTarget": []}
        for trial_data in self.subjects_trials_data[subject_id]:
            gt = trial_data["groundTruth"]
            rt = trial_data["responseTime"]
            rts[gt].append(rt)

        gt_types = [""] + [t for t in rts.keys()] + [""]
        rts_plot = [v for v in rts.values()]

        plt.figure("RT boxplot")
        plt.clf()
        plt.boxplot(rts_plot, showfliers=False, showmeans=True)
        plt.ylim([0.4, 2])
        plt.xticks(list(range(len(gt_types))), gt_types)
        plt.xlabel("Conditions")
        plt.ylabel("Response times")
        plt.pause(0.01)
        plt.savefig("{}/rt_boxplot/{}_{}.png".format(self.figures_dir, subject_id, state))

    def __time_frequency_analysis(self):
        if self.feature_type == "stft":
            self.__calculate_stft_spectrum()
        elif self.feature_type == "wavelet":
            self.__calculate_wavelet_spectrum()

    def __calculate_stft_spectrum(self):
        for subject_id, trials_data in self.subjects_trials_data.items():
            progressbar_prefix = "Calculating STFT spectrum, {}".format(subject_id)
            for trial_data in tqdm_info(trials_data, prefix=progressbar_prefix):
                spectrum = []
                for signal_single_channel in trial_data["eeg"].T:
                    if self.data_type == "session":
                        signal_single_channel = signal_single_channel[
                                                self.session_signal_start * self.sample_rate: self.session_signal_end * self.sample_rate]
                    f, t, zxx = stft(signal_single_channel,
                                     fs=self.sample_rate,
                                     nperseg=self.stft_nperseg,
                                     noverlap=int(self.stft_nperseg * self.stft_noverlap_ratio))
                    f, zxx = freq_band_selection(f, zxx, min_freq=self.stft_min_freq, max_freq=self.stft_max_freq)
                    spectrum.append(np.abs(zxx))

                    if self.stft_visualize:
                        plt.figure("STFT")
                        plt.clf()
                        plt.pcolormesh(t, f, np.abs(zxx), shading="auto")
                        plt.title("STFT Magnitude")
                        plt.ylabel("Frequency [Hz]")
                        plt.xlabel("Time [sec]")
                        plt.waitforbuttonpress()

                trial_data["stft_spectrum"] = np.array(spectrum).transpose([1, 2, 0])

    def __calculate_wavelet_spectrum(self):
        pass
        # for subject_id, trials_data in self.subjects_trials_data.items():
        #     print_info("Calculating wavelet spectrum, {}".format(subject_id))
        #     for trial_data in trials_data:
        #         spectrum = []
        #         for signal_single_channel in trial_data["eeg"].T:
        #             spectrum.append()
        #         trial_data["wavelet_spectrum"] = np.array(spectrum).transpose([1, 2, 0])

    def get_subject_ids(self):
        return [os.path.basename(i) for i in glob_sorted(self.dataset_dir + "/*")]


def print_info(string):
    printf("[INFO] {}".format(string), color="green")


def print_warning(string):
    printf("[WARNING] {}".format(string), color="yellow", attrs=["bold"])


def printf(string, color="yellow", on_color=None, attrs=None, **kwargs):
    """
    color:
        "grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    on_color:
        "on_grey", "on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan", "on_white"
    attrs:
        "bold", "dark", "underline", "blink", "reverse", "concealed"
    """

    if "linux" not in platform.platform().lower():
        print(string)
        return

    available_colors = ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "random"]
    available_on_colors = ["on_grey", "on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan", "on_white", None]

    assert color in available_colors, "color must be in {}".format(available_colors)
    assert on_color in available_on_colors, "on_color must be in {}".format(available_on_colors)

    if color == "random":
        r = random.randint(0, len(available_colors) - 2)
        color = available_colors[r]

    # if attrs is None:
    #     attrs = ["bold"]

    cprint(string, color=color, on_color=on_color, attrs=attrs, **kwargs)


def custom_sort(my_list):
    def convert(text):
        return float(text) if text.isdigit() else text

    def alphanum(key):
        return [convert(c) for c in re.split("([-+]?[0-9]?[0-9]*)", key)]

    my_list.sort(key=alphanum)
    return my_list


def glob_sorted(path):
    return custom_sort(glob(path))


def load_npy(path):
    try:
        data = np.load(path, allow_pickle=True).item()
    except ValueError:
        data = np.load(path, allow_pickle=True)
    return data


def get_fatigue_level(file_dir, basis):
    high_fatigue_npy_path = None
    low_fatigue_npy_path = None

    npy_paths = [p for p in glob_sorted(file_dir + "/*.npy") if "rest" in p]

    if basis == "by_feedback":
        with open(file_dir + "/fatigue_level.txt", "r") as f:
            lines = f.readlines()

        high = re.split("[:\n]", lines[0])[1]
        low = re.split("[:\n]", lines[1])[1]
        high_fatigue_npy_path = npy_paths[int(high) - 1]
        low_fatigue_npy_path = npy_paths[int(low) - 1]

    elif basis == "by_time":
        high_fatigue_npy_path = npy_paths[-1]
        low_fatigue_npy_path = npy_paths[0]

    return high_fatigue_npy_path, low_fatigue_npy_path


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def freq_band_selection(frequencies, zxx_, min_freq, max_freq):
    selected_frequencies = []
    selected_zxx = []
    for nth_freq, (freq, z) in enumerate(zip(frequencies, zxx_)):
        if min_freq <= freq <= max_freq:
            selected_frequencies.append(freq)
            selected_zxx.append(z)

    return np.array(selected_frequencies), np.array(selected_zxx)


def sigmoid(x, x_scale, x_shift):
    return 1 / (1 + math.exp(-x_scale * (x - x_shift)))


def signal_sticking(sig):
    left = sig[0]
    right = sig[-1]
    left_sig = sig - (right - left)
    right_sig = sig - (left - right)
    return np.concatenate([left_sig, sig, right_sig], axis=0)


def butter_bandpass_filter_sosfilt(data, low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    y = sosfilt(sos, data)
    return y


def butter_bandpass_filter_filtfilt(data, low_cut, high_cut, fs, order=5):
    wn1 = 2 * low_cut / fs
    wn2 = 2 * high_cut / fs
    [b, a] = butter(order, [wn1, wn2], btype="bandpass", output="ba")
    y = filtfilt(b, a, data)
    return y


def tqdm_info(i, prefix=""):
    time.sleep(0.1)
    prefix = "[INFO] " + prefix
    return tqdm(i, desc=prefix)


def plt_show_full_screen():
    # this function must be called before plt.show()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


if __name__ == "__main__":
    loader = DatasetLoader()
    subjects_trials_data, reformatted_data = loader.load_data(data_type="session",
                                                              feature_type="time",
                                                              # selected_channels=["C3", "Cz", "C4"],
                                                              )
    print("Done")