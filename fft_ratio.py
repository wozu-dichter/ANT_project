import matplotlib.pyplot as plt
import numpy as np
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, signal_sticking
from freqency_train import fft_call_cnn_model, normalize, plot_acc_val
from train_model import ConfusionMatrix
# from scipy.fftpack import fft
from scipy.signal import filtfilt as fft
from scipy.signal import butter


def get_fft_band(freqs, data):
    delta = data[(np.where((freqs > 0) & (freqs <= 4)))].sum(axis=0)
    theta = data[(np.where((freqs > 4) & (freqs <= 7)))].sum(axis=0)
    alpha = data[(np.where((freqs > 7) & (freqs <= 13)))].sum(axis=0)
    beta = data[(np.where((freqs > 13) & (freqs <= 30)))].sum(axis=0)

    return alpha, theta, beta, delta


def comput_psd_ratio(subject_band_array):
    alpha_high = subject_band_array[:, :, 0]
    theta_high = subject_band_array[:, :, 1]
    beta_high = subject_band_array[:, :, 2]
    delta_high = subject_band_array[:, :, 3]

    alpha_low = subject_band_array[:, :, 4]
    theta_low = subject_band_array[:, :, 5]
    beta_low = subject_band_array[:, :, 6]
    delta_low = subject_band_array[:, :, 7]

    ratio_1_high = ((alpha_high + theta_high) / beta_high)
    ratio_1_low = ((alpha_low + theta_low) / beta_low)
    ratio_2_high = (alpha_high / beta_high)
    ratio_2_low = (alpha_low / beta_low)
    ratio_3_high = ((delta_high + theta_high) / (alpha_high + beta_high))
    ratio_3_low = ((delta_low + theta_low) / (alpha_low + beta_low))
    ratio_4_high = (theta_high / beta_high)
    ratio_4_low = (theta_low / beta_low)

    ratio_1_high = np.array(ratio_1_high).mean(axis=1)
    ratio_1_low = np.array(ratio_1_low).mean(axis=1)
    ratio_2_high = np.array(ratio_2_high).mean(axis=1)
    ratio_2_low = np.array(ratio_2_low).mean(axis=1)
    ratio_3_high = np.array(ratio_3_high).mean(axis=1)
    ratio_3_low = np.array(ratio_3_low).mean(axis=1)
    ratio_4_high = np.array(ratio_4_high).mean(axis=1)
    ratio_4_low = np.array(ratio_4_low).mean(axis=1)

    return ratio_1_high, ratio_1_low, ratio_2_high, ratio_2_low, ratio_3_high, ratio_3_low, ratio_4_high, ratio_4_low


def plot_ratio(ratio_1_high, ratio_1_low, ratio_2_high, ratio_2_low, ratio_3_high, ratio_3_low, ratio_4_high,
               ratio_4_low, r_1_h_nor, r_1_l_nor, r_2_h_nor, r_2_l_nor, r_3_h_nor, r_3_l_nor, r_4_h_nor, r_4_l_nor):
    n = np.arange(0, ratio_1_high.shape[0])
    plt.figure(figsize=(15, 15))
    plt.subplot(421)
    plt.plot(n, ratio_1_high, label="fatigue")
    plt.plot(n, ratio_1_low, label="no fatigue")
    plt.title(r'$(\alpha + \theta)/ \beta$')
    plt.ylabel("ratio")
    plt.text(x=20, y=0.1, s='fatige:' + str(ratio_1_high.sum()) + ', no fatigue:' + str(ratio_1_low.sum()))
    plt.tight_layout()
    plt.legend()

    plt.subplot(423)
    plt.plot(n, ratio_2_high, label="fatigue")
    plt.plot(n, ratio_2_low, label="no fatigue")
    plt.text(x=20, y=0.1, s='fatige:' + str(ratio_2_high.sum()) + ', no fatigue:' + str(ratio_2_low.sum()))
    plt.title(r'$\alpha / \beta$')
    plt.tight_layout()
    plt.legend()

    plt.subplot(425)
    plt.plot(n, ratio_3_high, label="fatigue")
    plt.plot(n, ratio_3_low, label="no fatigue")
    plt.title(r'$(\delta + \theta)/(\alpha + \beta)$')
    plt.text(x=20, y=0.1, s='fatige:' + str(ratio_3_high.sum()) + ', no fatigue:' + str(ratio_3_low.sum()))
    plt.xlabel("sample point")
    plt.tight_layout()
    plt.legend()

    plt.subplot(427)
    plt.plot(n, ratio_4_high, label="fatigue")
    plt.plot(n, ratio_4_low, label="no fatigue")
    plt.title(r'$\theta / \beta$')
    plt.text(x=20, y=0.1, s='fatige:' + str(ratio_4_high.sum()) + ', no fatigue:' + str(ratio_4_low.sum()))
    plt.tight_layout()
    plt.legend()

    plt.subplot(422)
    plt.plot(n, r_1_h_nor, label="fatigue")
    plt.plot(n, r_1_l_nor, label="no fatigue")
    plt.title('normlize: ' + r'$(\alpha + \theta)/ \beta$')
    # plt.text(x=20, y=0.1, s='fatige:' + str(r_1_h_nor.sum()) + ', no fatigue:' + str(r_1_l_nor.sum()))
    plt.tight_layout()
    plt.legend()

    plt.subplot(424)
    plt.plot(n, r_2_h_nor, label="fatigue")
    plt.plot(n, r_2_l_nor, label="no fatigue")
    # plt.text(x=20, y=0.1, s='fatige:' + str(r_2_h_nor.sum()) + ', no fatigue:' + str(r_2_l_nor.sum()))
    plt.title('normlize: ' + r'$\alpha / \beta$')
    plt.tight_layout()
    plt.legend()

    plt.subplot(426)
    plt.plot(n, r_3_h_nor, label="fatigue")
    plt.plot(n, r_3_l_nor, label="no fatigue")
    plt.title('normlize: ' + r'$(\delta + \theta)/(\alpha + \beta)$')
    # plt.text(x=20, y=0.1, s='fatige:' + str(r_3_h_nor.sum()) + ', no fatigue:' + str(r_3_l_nor.sum()))
    plt.xlabel("sample point")
    plt.tight_layout()
    plt.legend()

    plt.subplot(428)
    plt.plot(n, r_4_h_nor, label="fatigue")
    plt.plot(n, r_4_l_nor, label="no fatigue")
    plt.title('normlize: ' + r'$\theta / \beta$')
    # plt.text(x=20, y=0.1, s='fatige:' + str(r_4_h_nor.sum()) + ', no fatigue:' + str(r_4_l_nor.sum()))
    plt.tight_layout()
    plt.legend()


def plot_psd_ratio(subject_band_array, subject_band_array_nor, key, path='./train_weight/fft_ratio/'):
    print(key)
    ratio_1_high, ratio_1_low, ratio_2_high, ratio_2_low, ratio_3_high, ratio_3_low, ratio_4_high, ratio_4_low = comput_psd_ratio(
        subject_band_array)
    r_1_h_nor, r_1_l_nor, r_2_h_nor, r_2_l_nor, r_3_h_nor, r_3_l_nor, r_4_h_nor, r_4_l_nor = comput_psd_ratio(
        subject_band_array_nor)

    plot_ratio(ratio_1_high, ratio_1_low, ratio_2_high, ratio_2_low, ratio_3_high, ratio_3_low, ratio_4_high,
               ratio_4_low, r_1_h_nor, r_1_l_nor, r_2_h_nor, r_2_l_nor, r_3_h_nor, r_3_l_nor, r_4_h_nor, r_4_l_nor)

    file_path = path + key + '/'
    plt.suptitle(key + ' : FFT ')
    # plt.savefig(file_path + key + '_' + eeg_channel[i_channel] + '.png')
    plt.savefig(file_path + key + '_compared_norm_minus' + '.png')
    plt.clf()
    plt.close()


def get_baseline_fft(data, subject_id, data_normalize=False):
    sampling_rate = 512
    all_alpha_array = []
    all_theta_array = []
    all_beta_array = []
    all_delta_array = []

    all_alpha_array_nor = []
    all_theta_array_nor = []
    all_beta_array_nor = []
    all_delta_array_nor = []
    for i in range(0, 60, 5):
        raw_data = data['eeg'][i:i + 5, :, :].reshape(-1, 32)
        subject_array = fft_compute(raw_data.T)
        subject_array_nor = fft_compute(np.array([normalize(i) for i in raw_data.T]))
        freqs = np.linspace(1, high_fre, subject_array.shape[0])

        alpha, theta, beta, delta = get_fft_band(freqs, subject_array)
        alpha_nor, theta_nor, beta_nor, delta_nor = get_fft_band(freqs, subject_array_nor)  # normal
        all_alpha_array.append(alpha)
        all_theta_array.append(theta)
        all_beta_array.append(beta)
        all_delta_array.append(delta)

        all_alpha_array_nor.append(alpha_nor)
        all_theta_array_nor.append(theta_nor)
        all_beta_array_nor.append(beta_nor)
        all_delta_array_nor.append(delta_nor)

    alpha_array = np.array(all_alpha_array).mean(axis=0)  # 對12筆資料做平均
    theta_array = np.array(all_theta_array).mean(axis=0)
    beta_array = np.array(all_beta_array).mean(axis=0)
    delta_array = np.array(all_delta_array).mean(axis=0)

    alpha_array_nor = np.array(all_alpha_array_nor).mean(axis=0)  # 對12筆資料做平均
    theta_array_nor = np.array(all_theta_array_nor).mean(axis=0)
    beta_array_nor = np.array(all_beta_array_nor).mean(axis=0)
    delta_array_nor = np.array(all_delta_array_nor).mean(axis=0)

    return alpha_array, theta_array, beta_array, delta_array, alpha_array_nor, theta_array_nor, beta_array_nor, delta_array_nor


def fft_compute(array):
    sampling_rate = 512
    wn1 = 2 * loader.bandpass_low_cut / sampling_rate
    wn2 = 2 * loader.bandpass_high_cut / sampling_rate
    [b, a] = butter(loader.bandpass_filter_order, [wn1, wn2], btype="bandpass", output="ba")
    fft_array = [(abs(fft(b, a, signal_sticking(i))) / sampling_rate)[:len(abs(fft(b, a, signal_sticking(i)))) // 2] for
                 i in array]
    fft_array = np.array(fft_array).T
    subject_array = fft_array[1:int((high_fre / (sampling_rate // 2)) * len(fft_array)), :]
    return subject_array


eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

sampling_rate = 512
high_fre = 30
loader = DatasetLoader()
loader.apply_signal_normalization = False

subject_ids = loader.get_subject_ids()

subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type='time',
                                           # single_subject=id,
                                           fatigue_basis='by_feedback'
                                           )

for key in subject_ids:
    eeg_subject = subjects_trials_data[key]
    subject_band_array_high = []
    subject_band_array_low = []
    subject_band_array_high_nor = []
    subject_band_array_low_nor = []

    baseline_path = './dataset2/' + key + '/1/*.npy'
    baseline_path = [p for p in glob_sorted(baseline_path) if 'baseline' in p][0]
    baseline_data = load_npy(baseline_path)
    alpha_array, theta_array, beta_array, delta_array, alpha_array_nor, theta_array_nor, beta_array_nor, delta_array_nor = get_baseline_fft(
        baseline_data, key, data_normalize=loader.apply_signal_normalization)
    band_array = []
    band_array_nor = []
    for data in eeg_subject:
        if data['fatigue_level'] == 'high' or 'low':
            subject_array = data['eeg'].T
            subject_array = fft_compute(data['eeg'].T)  # get FFT output shape[149,32]
            subject_array_nor = fft_compute(np.array([normalize(i) for i in data['eeg'].T]))
            freqs = np.linspace(1, high_fre, subject_array.shape[0])
            alpha, theta, beta, delta = get_fft_band(freqs, subject_array)
            alpha_nor, theta_nor, beta_nor, delta_nor = get_fft_band(freqs, subject_array_nor)  # normalize

            # band_array = (
            #     [alpha - alpha_array, theta - theta_array, beta - beta_array,
            #      delta - delta_array])
            # band_array_nor = ([alpha_nor - alpha_array, theta_nor - theta_array,
            #                        beta_nor - beta_array, delta_nor - delta_array])
            band_array = ([alpha, theta, beta, delta])
            band_array_nor = ([alpha_nor, theta_nor, beta_nor, delta_nor])
            if data['fatigue_level'] == 'high':
                subject_band_array_high.append(np.array(band_array))
                subject_band_array_high_nor.append(np.array(band_array_nor))
            elif data['fatigue_level'] == 'low':
                subject_band_array_low.append(np.array(band_array))
                subject_band_array_low_nor.append(np.array(band_array_nor))

    subject_band_array_high = np.array(subject_band_array_high)
    subject_band_array_low = np.array(subject_band_array_low)
    subject_band_array = np.concatenate((subject_band_array_high, subject_band_array_low), axis=1).transpose([0, 2, 1])
    subject_band_array_nor = np.concatenate((subject_band_array_high_nor, subject_band_array_low_nor),
                                            axis=1).transpose([0, 2, 1])

    plot_psd_ratio(subject_band_array, subject_band_array_nor, key)  # plot everyone eeg band ratio
