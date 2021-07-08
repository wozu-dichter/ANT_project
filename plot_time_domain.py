from ANT_dataset_loader import DatasetLoader
import numpy as np
from matplotlib import pyplot as plt
import os


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def plot_dataset_X(dataset_X, figure_name):
    pick_chaneel = np.array([6, 8, 10, 11, 12, 15])
    time_domaim = dataset_X[pick_chaneel, :]  # C3, C4, P3, Pz, P4, Oz
    # eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7",
    #                "C3", "Cz", "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz"]
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]
    rows = pick_chaneel.shape[0]
    time = np.linspace(0, 5, 512 * 5)
    fig = plt.figure(figure_name, figsize=(20, 16))
    for i in range(32):
        print(i)
        plt.subplot(16, 2, i + 1)
        # plt.title(eeg_channel[pick_chaneel[i]], fontsize=10)
        plt.title(eeg_channel[i], fontsize=10)
        # plt.plot(time, time_domaim[i, :])
        plt.plot(time, dataset_X[i, :])
        plt.tight_layout()
    plt.xlabel('Time (Hz)')

    plt.savefig("./train_weight/time_domain/" + figure_name + ".png")
    plt.close()


def plot_dataset_all_people(dataset_X, channel_name, figure_name):
    time = np.linspace(0, 5, 512 * 5)
    plt.figure(num=3, figsize=(20, 16))
    plt.clf()
    # plt.ion()
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.title(subject_ids[i], fontsize=10, loc='left')
        plt.plot(time, dataset_X[i, :])
        plt.tight_layout()
        ymax = max(dataset_X[i, :])
        xpos = np.where(dataset_X[i, :] == ymax)[0]
        xmax = time[xpos]
        # plt.annotate("max:{:.3f}".format(ymax), xy=[xmax, ymax])

        ymin = min(dataset_X[i, :])
        xpos = np.where(dataset_X[i, :] == ymin)[0]
        xmin = time[xpos]
        # plt.annotate("min:{:.3f}".format(ymin), xy=[xmin, ymin])
        # plt.annotate("abs:{:.3f}".format(abs(ymax - ymin)), xy=[0, ymin])
        plt.text(0, ymin, "max:{:.3f},\nmin:{:.3f},\nabs:{:.3f}".format(ymax, ymin, abs(ymax - ymin)), color='red')

    plt.show()
    figure_name = channel_name + '_' + figure_name
    plt.suptitle(figure_name)
    create_dir("./train_weight/time_domain/" + channel_name + '/')
    plt.savefig("./train_weight/time_domain/" + channel_name + '/' + figure_name + ".png")
    # plt.close()


if __name__ == '__main__':
    plot_what = 'one_channel_with_all'  # '32channel_with_each' / 'one_channel_with_all'
    loader = DatasetLoader()
    loader.rest_signal_len = 5
    subject_ids = loader.get_subject_ids()
    loader.apply_signal_normalization = False
    loader.apply_bandpass_filter = True

    if loader.apply_signal_normalization == True:
        loader.normalization_mode = "mean_norm"
    if plot_what == '32channel_with_each':
        for subject_id in subject_ids:
            subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                                       single_subject=subject_id,
                                                       fatigue_basis="by_time")
            for key, subjects_data in subjects_trials_data.items():
                hight_num, low_num = 0, 0
                for index in subjects_data:
                    if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                        subject_array = index['eeg'].T
                        if index['fatigue_level'] == 'high':
                            figure_name = key + "_high_" + str(hight_num)
                            hight_num += 1
                        elif index['fatigue_level'] == 'low':
                            figure_name = key + "_low_" + str(low_num)
                            low_num += 1

                        plot_dataset_X(subject_array, figure_name)
    elif plot_what == 'one_channel_with_all':
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                                   fatigue_basis="by_time")
        new_eeg_data_high = {}
        new_eeg_data_low = {}
        for key, value in subjects_trials_data.items():
            high_array = []
            low_array = []
            for i_index in value:
                if i_index['fatigue_level'] == 'high':
                    high_array.append(i_index['eeg'])
                elif i_index['fatigue_level'] == 'low':
                    low_array.append(i_index['eeg'])
            new_eeg_data_high.update({key: high_array})
            new_eeg_data_low.update({key: low_array})

        del key, value, low_array, high_array, i_index

        eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                       "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                       "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                       "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

        for i in range(60):
            plot_high_array = []
            plot_low_array = []
            for subject_id in subject_ids:
                some_one_array = new_eeg_data_high[subject_id][i]
                plot_high_array.append(some_one_array)
                some_one_array = new_eeg_data_low[subject_id][i]
                plot_low_array.append(some_one_array)
            for i_channel in range(32):
                channel_name = eeg_channel[i_channel]
                plot_array_high = []
                plot_array_low = []
                for index_array in plot_low_array:
                    plot_array_high.append(index_array[:, i_channel])  # plot high
                for index_array in plot_high_array:
                    plot_array_low.append(index_array[:, i_channel])

                plot_dataset_all_people(np.array(plot_array_high), channel_name=channel_name,
                                        figure_name='high_' + str(i))
                plot_dataset_all_people(np.array(plot_array_low), channel_name=channel_name,
                                        figure_name='low_' + str(i))

""""
eeg_data = np.load('eeg_data.npy')
eeg_label = np.load('eeg_label.npy')
eeg_label = argmax(eeg_label, axis=1)  # reverse to_categorical

df = pd.read_csv('channel(10-20).txt')
ch_names = df.name.to_list()

info = mne.create_info(
    ch_names=ch_names[:16],
    ch_types=['eeg']*eeg_data.shape[0],
    sfreq=500
)

custom_raw = mne.io.RawArray(eeg_data, info)
picks = mne.pick_channels(custom_raw.info['ch_names'],
                              ['MEG 2443', 'MEG 2442', 'MEG 2441'])
"""
