import numpy as np
from scipy.signal import stft, butter, sosfilt, filtfilt
from EEGModels import EEGNet_TINA_TEST
from tensorflow.keras.models import load_model

def preprocessing(raw_data):
    # raw_data: input_shape=[time_points, channel_nums] ex:[2500, 16]
    # fix input_shape if not input_shape=[time_points, channel_nums]
    if raw_data.shape[0] < raw_data.shape[1]:
        raw_data = raw_data.T

    ############################## band_pass_filter  ##############################
    bandpass_filter_order = 1
    bandpass_low_cut = 5
    bandpass_high_cut = 28
    sample_rate = 500
    for i_channel in range(raw_data.shape[1]):
        filtered_signal_copied = butter_bandpass_filter(data=raw_data[:, i_channel],
                                                        low_cut=bandpass_low_cut,
                                                        high_cut=bandpass_high_cut,
                                                        fs=sample_rate,
                                                        order=bandpass_filter_order)
        raw_data[:, i_channel] = raw_data[:, i_channel]
    ############################## ############################## ####################

    return raw_data

def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    wn1 = 2 * low_cut / fs
    wn2 = 2 * high_cut / fs
    [b, a] = butter(order, [wn1, wn2], btype="bandpass", output="ba")
    y = filtfilt(b, a, data)
    return y


def predict_fatigue(eeg_data):
    eeg_data = preprocessing(eeg_data)


# if __name__ == '__main__':
eeg_data = np.load('eeg_data.npy')[0, :, :]
predict_fatigue(eeg_data)
model = load_model('fatigue_predict.h5')