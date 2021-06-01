import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model


def preprocessing(raw_data, normalize_mode='None'):
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
        ################ normalize ###############
        if normalize_mode == 'min_max':
            raw_data[:, i_channel] = raw_data[:, i_channel] - np.min(raw_data[:, i_channel], axis=0)
            raw_data[:, i_channel] = raw_data[:, i_channel] / np.max(raw_data[:, i_channel], axis=0)
        elif normalize_mode == "z_score":
            raw_data[:, i_channel] = raw_data[:, i_channel] - np.mean(raw_data[:, i_channel], axis=0, dtype=np.float64)
            raw_data[:, i_channel] = raw_data[:, i_channel] / np.std(raw_data[:, i_channel], axis=0, dtype=np.float64)
        elif normalize_mode == "mean_norm":
            raw_data[:, i_channel] = raw_data[:, i_channel] - np.mean(raw_data[:, i_channel], axis=0, dtype=np.float64)

        filtered_signal_copied = butter_bandpass_filter(data=raw_data[:, i_channel],
                                                        low_cut=bandpass_low_cut,
                                                        high_cut=bandpass_high_cut,
                                                        fs=sample_rate,
                                                        order=bandpass_filter_order)
        raw_data[:, i_channel] = filtered_signal_copied
    ############################## ############################## ####################

    ###### reshape input data for model ##########
    chans, samples, kernels = raw_data.shape[1], raw_data.shape[0], 1
    x_test = raw_data.reshape(
        (1, chans, samples, kernels))  # reshape data to model shape [num of test , channel_num, samples , kernel]

    return x_test


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    wn1 = 2 * low_cut / fs
    wn2 = 2 * high_cut / fs
    [b, a] = butter(order, [wn1, wn2], btype="bandpass", output="ba")
    y = filtfilt(b, a, data)
    return y


def predict_fatigue(eeg_data):
    eeg_data = preprocessing(eeg_data, normalize_mode='min_max')
    scores = model.predict(eeg_data)[0]  # scores = [tired score, not tired score]
    return round(scores[0] * 10)


if __name__ == '__main__':
    import random

    eeg_data = np.load('eeg_data.npy')[random.randint(0, 2000), :,
               :]  # input example shape=[2500,16] or [16,2500] also acceptable

    model = load_model('fatigue_predict.h5')  # the model val_acc = 0.666 val_loss = 0.720
    fatigue_level = predict_fatigue(eeg_data)  # output range: min 0 to max 10
    print(fatigue_level)
