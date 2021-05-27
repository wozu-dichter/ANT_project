from ANT_dataset_loader import DatasetLoader
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def min_max(X):
    X = X - np.min(X, axis=0)
    min_max = X / np.max(X, axis=0)
    return min_max


def plot_dataset_X(dataset_X, figure_name):
    pick_chaneel = np.array([6, 8, 10, 11, 12, 15])
    time_domaim = dataset_X[pick_chaneel, :]  # C3, C4, P3, Pz, P4, Oz
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7",
                   "C3", "Cz", "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz"]
    rows = pick_chaneel.shape[0]

    fig = plt.figure(figure_name, figsize=(20, 16))
    for i in range(rows):
        fs, fre1, fre2 = 500, 5, 28
        wn1 = 2 * fre1 / fs
        wn2 = 2 * fre2 / fs
        b, a = signal.butter(5, [wn1, wn2], 'bandpass')  # 配置濾波器 8 表示濾波器的階數
        filtedData = signal.filtfilt(b, a, time_domaim[i, :2500])  # data為要過濾的訊號
        filtedData = min_max(filtedData)
        time = np.linspace(0, 5, 2500)
        plt.subplot(rows, 1, i + 1)
        plt.title(eeg_channel[pick_chaneel[i]], fontsize=10)
        # plt.plot(time_domaim[i, :2500], c='r', label="original eeg", linestyle="--")
        plt.plot(time, filtedData, c='b', label="band pass")
        plt.tight_layout()
    plt.xlabel('Time')
    plt.ylabel('Amp')
    plt.savefig("./train_weight/band_pass/" + figure_name + ".png")
    plt.close()


if __name__ == '__main__':
    loader = DatasetLoader()
    subject_ids = loader.get_subject_ids()
    for subject_id in subject_ids:
        subjects_trials_data = loader.load_data(data_type="rest", feature_type="time",
                                                single_subject=subject_id,
                                                fatigue_basis="by_time")
        for key, subjects_data in subjects_trials_data.items():
            hight_num = 0
            low_num = 0
            for index in subjects_data:

                if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                    subject_array = index['eeg'].T
                    if index['fatigue_level'] == 'high':
                        figure_name = key + "_high_bandpass_" +str(hight_num)
                        hight_num+=1
                    elif index['fatigue_level'] == 'low':
                        figure_name = key + "_low_bandpass_" + str(low_num)
                        low_num += 1
                    plot_dataset_X(subject_array, figure_name)
