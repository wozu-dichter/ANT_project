from ANT_dataset_loader import DatasetLoader
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from numpy import argmax

#plot time domain data

def plot_dataset_X(dataset_X, figure_name):
    pick_chaneel = np.array([6, 8, 10, 11, 12, 15])
    time_domaim = dataset_X[pick_chaneel, :]  # C3, C4, P3, Pz, P4, Oz
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7",
                   "C3", "Cz", "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz"]
    rows = pick_chaneel.shape[0]
    time = np.linspace(0, 5, 2500)
    fig = plt.figure(figure_name, figsize=(20, 16))
    for i in range(rows):
        plt.subplot(rows, 1, i + 1)
        plt.title(eeg_channel[pick_chaneel[i]], fontsize=10)
        plt.plot(time, time_domaim[i, :])
        plt.tight_layout()
    plt.xlabel('Time (Hz)')
    plt.savefig("./train_weight/time_domain/" + figure_name + ".png")
    plt.close()


if __name__ == '__main__':
    loader = DatasetLoader()
    subject_ids = loader.get_subject_ids()
    for subject_id in subject_ids:
        subjects_trials_data = loader.load_data(data_type="rest", feature_type="time",
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
