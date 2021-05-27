import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft
from scipy.fft import fft
from custom_lib import load_npy, glob_sorted


def get_file_paths(file_dir, data_type, suffix):
    paths = [p for p in glob_sorted(file_dir + "/*{}".format(suffix)) if data_type in p]
    return paths


def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)


def freq_band_selection(freqs, zxx_, min_freq, max_freq):
    selected_freqs = []
    selected_zxx = []
    for nth_freq, (freq, z) in enumerate(zip(freqs, zxx_)):
        if min_freq <= freq <= max_freq:
            selected_freqs.append(freq)
            selected_zxx.append(z)

    return np.array(selected_freqs), np.array(selected_zxx)


def apply_fft(data, sampling_rate):
    yf = (abs(fft(data)) / sampling_rate)[:sampling_rate // 2]
    xf = np.arange(len(data))[:sampling_rate // 2]

    return xf, yf


def apply_normalization(signal, mode="z_score"):
    available_modes = ["min_max", "z_score"]
    assert mode in available_modes

    if mode == "min_max":
        signal = signal - np.min(signal)
        signal = signal / np.max(signal)
    elif mode == "z_score":
        signal = signal - np.mean(signal)
        signal = signal / np.std(signal)

    return signal


dataset_dir = "./dataset"
channel_orders = ["Fp1", "Fp2", "F3", "Fz", "F4",
                  "T7", "C3", "Cz", "C4", "T8",
                  "P3", "Pz", "P4", "P7", "P8",
                  "Oz"]

num_channels = 16
sample_rate = 500
start_time = 2
end_time = 3
plt.ion()

for subject_dir in glob_sorted(dataset_dir + "/*"):
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = get_file_paths(file_dir=record_dir, data_type="session", suffix=".npy")
        for npy_path in npy_paths:
            npy_data = load_npy(npy_path)
            raw_eeg = npy_data["eeg"]
            multi_trial = npy_data["multiTrialData"]
            figure_dir = npy_path.replace(".npy", "")
            create_dir(figure_dir)
            for nth_channel, channel_name in enumerate(channel_orders):
                figure_dir_channel = "{}/{}".format(figure_dir, channel_name)
                create_dir(figure_dir_channel)

                for nth_trial, single_trial in multi_trial.items():
                    if not single_trial["hasAnswer"]:
                        continue
                    nth_trial -= 1
                    trial_eeg = raw_eeg[nth_trial * 4: nth_trial * 4 + 4].reshape((-1, num_channels))
                    response_eeg = trial_eeg[start_time * sample_rate:end_time * sample_rate]

                    # data = response_eeg[:, 0]
                    data = trial_eeg[:, nth_channel]
                    data = apply_normalization(data)

                    # wavelet transform
                    figure_dir_channel_wavelet = figure_dir_channel + "/wavelet"
                    create_dir(figure_dir_channel_wavelet)
                    plt.clf()
                    t = np.arange(0, 4, 1 / sample_rate)
                    wavename = "cgau8"
                    totalscal = 500
                    fc = pywt.central_frequency(wavename)
                    cparam = 2 * fc * totalscal
                    scales = cparam / np.arange(totalscal, 1, -1)
                    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sample_rate)
                    plt.figure(1, figsize=(8, 4))
                    plt.subplot(211)
                    plt.plot(t, data)
                    plt.xlabel("time(seconds)")
                    plt.subplot(212)

                    cut = 60
                    frequencies = frequencies[-cut:]
                    cwtmatr = cwtmatr[-cut:]

                    plt.contourf(t, frequencies, abs(cwtmatr), vmax=10)
                    plt.ylabel("frequency(Hz)")
                    plt.xlabel("time(seconds)")
                    plt.subplots_adjust(hspace=0.4)
                    plt.pause(0.001)
                    plt.show()
                    plt.savefig("{}/{}.png".format(figure_dir_channel_wavelet, nth_trial))

                    # # stft
                    figure_dir_channel_stft = figure_dir_channel + "/stft"
                    create_dir(figure_dir_channel_stft)
                    plt.clf()
                    plt.subplot(311)
                    plt.plot(data)

                    plt.subplot(312)
                    xf, yf = apply_fft(data, sample_rate)
                    plt.plot(xf, yf)

                    plt.subplot(313)
                    f, t, zxx = stft(data, fs=sample_rate, nperseg=100)
                    f, zxx = freq_band_selection(f, zxx, min_freq=0, max_freq=30)
                    plt.pcolormesh(t, f, np.abs(zxx), shading="auto")
                    plt.title("STFT Magnitude")
                    plt.ylabel("Frequency [Hz]")
                    plt.xlabel("Time [sec]")
                    plt.subplots_adjust(hspace=0.5)
                    plt.pause(0.001)
                    plt.show()
                    plt.savefig("{}/{}.png".format(figure_dir_channel_stft, nth_trial))

                    # welch
                    # figure_dir_channel_psd = figure_dir_channel + "/psd"
                    # create_dir(figure_dir_channel_psd)
                    # plt.clf()
                    # freq, psd = welch(data, fs=sample_rate, nperseg=sample_rate)
                    # plt.clf()
                    # plt.semilogx(freq, psd)
                    # plt.title("PSD: power spectral density")
                    # plt.xlabel("Frequency")
                    # plt.ylabel("Power")
                    # plt.tight_layout()
                    # plt.pause(0.001)
                    # plt.show()
                    # plt.savefig("{}/{}.png".format(figure_dir_channel_psd, nth_trial))
