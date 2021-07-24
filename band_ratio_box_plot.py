from ANT_dataset_loader import DatasetLoader
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt


def get_psd_band(freqs, psd):
    delta = psd[(np.where((freqs > 0) & (freqs <= 4)))].sum()
    theta = psd[(np.where((freqs > 4) & (freqs <= 7)))].sum()
    alpha = psd[(np.where((freqs > 7) & (freqs <= 13)))].sum()
    beta = psd[(np.where((freqs > 13) & (freqs <= 30)))].sum()

    return alpha, theta, beta, delta


def plot_box(low_array, high_array, showfliers=False, showmeans=True, whis=None):
    box_array = [np.array(high_array)[:, i] for i in range(len(eeg_channel))]
    plt.boxplot(box_array, labels=eeg_channel, showmeans=showmeans, showfliers=showfliers, boxprops={'color': 'b'},
                meanprops={"markeredgecolor": 'b', 'markerfacecolor': 'b', 'marker': 'o'},
                medianprops={'linestyle': '-', 'color': 'b'}, capprops={'color': 'b'},
                whiskerprops={'linestyle': '-', 'color': 'b'}, widths=0.25, whis=whis)

    box_array = [np.array(low_array)[:, i] for i in range(len(eeg_channel))]
    plt.boxplot(box_array, labels=eeg_channel, showmeans=showmeans, showfliers=showfliers, boxprops={'color': 'y'},
                meanprops={"markeredgecolor": 'y', 'markerfacecolor': 'y', 'marker': 'o'},
                medianprops={'linestyle': ':', 'color': 'y'}, capprops={'color': 'y'},
                whiskerprops={'linestyle': ":", 'color': 'y'}, whis=whis)


def plot_relative(file_name, low_fatigue, high_fatigue, path='./train_weight/psd_ratio/'):
    if find_outlier:
        showfliers = False  # True = display Outlier with boxplot
        whis = [0, 100]
    else:
        showfliers = True
        whis = None

    if file_name == 'boxplot_power_band_relative':
        y_label = [r'$\alpha$', r'$\theta$', r'$\beta$', r'$\delta$']

    else:
        y_label = [r'$(\alpha + \theta)/ \beta$', r'$\alpha / \beta$', r'$(\alpha + \theta)/(\beta+\theta)$',
                   r'$\theta / \beta$']

    plt.figure(file_name, figsize=(15, 15))

    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plot_box([a[:, i] for _, a in low_fatigue.items()],
                 [a[:, i] for _, a in high_fatigue.items()],
                 showfliers=showfliers, whis=whis)
        plt.ylabel(y_label[i])
        plt.plot([], c='b', label='high fatigue')
        plt.plot([], c='y', label='low fatigue')
        plt.legend()
        plt.tight_layout()
    plt.savefig(path + file_name + '.png')

    plt.figure('No outlier ' + file_name, figsize=(15, 15))
    plt.clf()
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plot_box([a[:, i] for _, a in low_fatigue.items()],
                 [a[:, i] for _, a in high_fatigue.items()],
                 showfliers=False, showmeans=False, whis=whis)
        plt.ylabel(y_label[i])
        plt.plot([], c='b', label='high fatigue')
        plt.plot([], c='y', label='low fatigue')
        plt.legend()
        plt.tight_layout()
    plt.savefig(path + 'NO_OUTLIER' + '.png')


def plot_error_bar(low_array, high_array):
    box_array = [np.array(high_array)[:, i] for i in range(len(eeg_channel))]
    plt.errorbar(x=eeg_channel, y=np.array(box_array).mean(axis=1), yerr=np.array(box_array).std(axis=1), capsize=4,
                 linestyle='-', marker='^', ecolor='b', color='b', label='high fatigue')

    box_array = [np.array(low_array)[:, i] for i in range(len(eeg_channel))]
    plt.errorbar(x=eeg_channel, y=np.array(box_array).mean(axis=1), yerr=np.array(box_array).std(axis=1), capsize=4,
                 linestyle=':', marker='^', ecolor='y', color='y', label='low fatigue')
    plt.legend()


def plot_errorbar(file_name, low_fatigue, high_fatigue, path='./train_weight/psd_ratio/'):
    if file_name == 'errorbar_power_band_relative':
        y_label = [r'$\alpha$', r'$\theta$', r'$\beta$', r'$\delta$']

    else:
        y_label = [r'$(\alpha + \theta)/ \beta$', r'$\alpha / \beta$', r'$(\alpha + \theta)/(\beta+\theta)$',
                   r'$\theta / \beta$']

    plt.figure(file_name, figsize=(15, 15))
    plt.clf()
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plot_error_bar([a[:, i] for _, a in low_fatigue.items()], [a[:, i] for _, a in high_fatigue.items()])
        plt.ylabel(y_label[i])
        plt.tight_layout()
    plt.savefig(path + file_name + '.png')


def compute_ratio(array):
    ratio_1 = np.nanmean((array[:, :, 1] + array[:, :, 0]) / array[:, :, 2],
                         axis=0)  # someone ratio average , output shape=[channel_num]
    ratio_2 = np.nanmean(array[:, :, 0] / array[:, :, 2], axis=0)
    ratio_3 = np.nanmean((array[:, :, 0] + array[:, :, 1]) / (array[:, :, 1] + array[:, :, 2]), axis=0)
    ratio_4 = np.nanmean(array[:, :, 1] / array[:, :, 2], axis=0)
    return [ratio_1, ratio_2, ratio_3, ratio_4]


def remove_outlier(array):
    n = 1
    outlier_filter_array = []
    for i_channel in range(array.shape[1]):
        channel_array = array[:, i_channel, :]
        # IQR = Q3-Q1
        IQR = np.percentile(channel_array, 75, axis=0) - np.percentile(channel_array, 25, axis=0)
        # outlier = Q3 + n*IQR
        outlier_big_index = [channel_array < np.percentile(channel_array, 75, axis=0) + n * IQR]
        # outlier = Q1 - n*IQR
        outlier_small_index = [channel_array > np.percentile(channel_array, 25, axis=0) - n * IQR]
        outlier_filter = np.array(outlier_big_index or outlier_small_index)[0]
        outlier_filter_array.append(outlier_filter)

    outlier_filter_array = np.array(outlier_filter_array)
    for index in range(outlier_filter_array.shape[1]):
        index_array = outlier_filter_array[:, index, :]
        for j in range(len(eeg_channel)):
            if False in index_array[j]:
                outlier_filter_array[j, index, :] = False
                array[index, j, :] = np.nan

    outlier_filter_array = outlier_filter_array.transpose(1, 0, 2)  # index where False = outliers
    return array


if __name__ == "__main__":
    sampling_rate = 512

    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

    # eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
    #                "C4", "T8", "P3", "Pz", "P4", "P7", "P8",
    #                "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
    #                "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", ]
    # eeg_channel = ['Fp1', 'F3', 'FC3', 'C3', 'C4', 'P3', 'T7']

    find_outlier = True
    loader = DatasetLoader()
    loader.apply_signal_normalization = False
    loader.apply_bandpass_filter = True

    subject_ids = loader.get_subject_ids()

    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type='time',
                                               fatigue_basis='by_feedback',
                                               selected_channels=eeg_channel,
                                               # single_subject="c95hyt",
                                               # excluded_subjects=['c95hyw']
                                               )
    low_fatigue = {}
    high_fatigue = {}
    all_low_ratio = {}
    all_high_ratio = {}

    for subject_id, array in subjects_trials_data.items():
        personal_low_fatigue = []
        personal_high_fatigue = []
        for record_index, record_array in array.items():
            for trial_array in record_array["trials_data"]:
                if trial_array["fatigue_level"] != None:
                    channel_psd = []
                    for i_channel in trial_array["eeg"].T:
                        freqs, psd = signal.welch(i_channel, fs=sampling_rate, nperseg=sampling_rate)
                        alpha, theta, beta, delta = get_psd_band(freqs, psd)
                        channel_psd.append([alpha, theta, beta, delta])
                    if trial_array["fatigue_level"] == "low":
                        personal_low_fatigue.append(np.array(channel_psd))
                        # low_ratio.append(np.array(channel_psd))
                    elif trial_array["fatigue_level"] == "high":
                        personal_high_fatigue.append(np.array(channel_psd))
                        # high_ratio.append(np.array(channel_psd))
        if find_outlier:
            personal_high_fatigue = remove_outlier(np.array(personal_high_fatigue)) # input shape [N, channel num ,4]
            personal_low_fatigue = remove_outlier(np.array(personal_low_fatigue))

        average_high_fatigue = np.nanmean(personal_high_fatigue, axis=0)
        average_low_fatigue = np.nanmean(personal_low_fatigue, axis=0)

        high_ratio = compute_ratio(np.array(personal_high_fatigue))
        low_ratio = compute_ratio(np.array(personal_low_fatigue))

        high_fatigue.update({subject_id: average_high_fatigue})
        low_fatigue.update({subject_id: average_low_fatigue})
        all_high_ratio.update({subject_id: np.array(high_ratio).T})
        all_low_ratio.update({subject_id: np.array(low_ratio).T})

    print('plot figure')
    plot_relative(file_name='boxplot_power_band_relative', low_fatigue=low_fatigue, high_fatigue=high_fatigue)
    plot_errorbar(file_name='errorbar_power_band_relative', low_fatigue=low_fatigue, high_fatigue=high_fatigue)

    plot_relative(file_name='boxplot_power_band_ratio', low_fatigue=all_low_ratio, high_fatigue=all_high_ratio)
    plot_errorbar(file_name='errorbar_power_band_ratio', low_fatigue=all_low_ratio, high_fatigue=all_high_ratio)

    plt.pause(0)
