import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, freq_band_selection, signal_sticking
from freqency_train import call_cnn_model, normalize, plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from train_model import ConfusionMatrix
import time
from scipy.signal import butter, filtfilt


def get_baseline_average_stft_eeg(data, subject_id, get_middle_value=False, data_normalize=False):
    ############ get baselinestft eeg(average) #############
    stft_nperseg = 500
    stft_noverlap_ratio = 0.95
    stft_min_freq = 3
    stft_max_freq = 28
    sample_rate = 512

    all_stft_array = []
    num = 0
    for i in range(0, 60, 5):
        stft_array = []

        raw_data = data['eeg'][i:i + 5, :, :].reshape(-1, 32)
        if loader.apply_bandpass_filter:
            for index in range(raw_data.shape[-1]):
                filtered_signal_copied = butter_bandpass_filter_filtfilt(data=signal_sticking(raw_data[:, index]),
                                                                         low_cut=loader.bandpass_low_cut,
                                                                         high_cut=loader.bandpass_high_cut,
                                                                         fs=loader.sample_rate,
                                                                         order=loader.bandpass_filter_order)
                start = int(len(filtered_signal_copied) * (1 / 3))
                end = int(len(filtered_signal_copied) * (2 / 3))
                filtered_signal = filtered_signal_copied[start:end]
                raw_data[:, index] = filtered_signal

        for ith_channel in range(raw_data.shape[1]):
            channel_data = raw_data[:, ith_channel]
            if data_normalize:
                channel_data = normalize(channel_data)
            ############stft compute###############
            f, t, zxx = stft(channel_data,
                             fs=sample_rate,
                             nperseg=stft_nperseg,
                             noverlap=int(stft_nperseg * stft_noverlap_ratio))
            selected_time = []
            selected_zxx = []
            ################ get stft middle value #################
            if get_middle_value:
                for i in range(t.shape[0]):  # get stft middle value
                    if t[i] < (len(channel_data) / sample_rate - 0.5) and t[i] > 0.5:
                        selected_time.append(t[i])
                        selected_zxx.append(zxx[:, i])
                t = np.array(selected_time)
                zxx = np.array(selected_zxx).T
            ##########################################
            f, zxx = freq_band_selection(f, abs(zxx), min_freq=stft_min_freq, max_freq=stft_max_freq)
            stft_array.append(zxx)
            #####################################
        num = num + 1
        all_stft_array.append(np.array(stft_array))

    all_stft_array = np.array(all_stft_array).mean(axis=0)
    return t, f, all_stft_array


def plot_baseline_stft(all_stft_array, t, f, subject_id):
    for ith_channel in range(all_stft_array.shape[0]):  # save average stft picture
        plt.pcolormesh(t, f, np.abs(all_stft_array[ith_channel, :, :]), vmin=-2, vmax=10, shading='auto')
        plt.title(eeg_channel[ith_channel])
        file_name = "/avg_" + subject_id + '_' + eeg_channel[ith_channel]
        plt.savefig("./train_weight/baseline/" + subject_id + file_name)
        plt.waitforbuttonpress()
        plt.clf()


def plot_minus_stft(key, minus_subject_array, baseline_eeg, num, get_middle_value):
    print('not actually fix t and fre')
    t=[]
    f=[]
    for i in range(minus_subject_array.shape[2]):
        plt.subplot(311)
        plt.pcolormesh(t, f, np.abs(minus_subject_array[:, :, i]+baseline_eeg[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title(key + ':raw data in ' + eeg_channel[i])
        plt.colorbar()
        plt.subplot(312)
        plt.pcolormesh(t, f, np.abs(baseline_eeg[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('baseline')
        plt.colorbar()
        plt.subplot(313)
        plt.pcolormesh(t, f, np.abs(minus_subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('after minus')
        plt.colorbar()
        if get_middle_value:
            plt.savefig("./train_weight/minus_stft_middle/" + key + "/" + key + '_' + eeg_channel[i] + '_' + num)
        else:
            plt.savefig("./train_weight/minus_stft/" + key + "/" + key + '_' + eeg_channel[i] + '_' + num)
        plt.clf()


def process_stft_subjects_data(subjects_trials_data, minus_stft_visualize):
    eeg_data = []
    eeg_label = []

    for key, subjects_data in subjects_trials_data.items():  # c95ccy
        high_num = 1
        low_num = 1
        for record_index, record_data in subjects_data.items():  # record0
            for index in record_data['trials_data']:
                if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                    subject_array = index['stft_baseline_removed']
                    eeg_data.append(subject_array)
                    if index['fatigue_level'] == 'high':  # tired
                        eeg_label.append(0)
                        if minus_stft_visualize:
                            num = 'high_' + str(high_num)
                            plot_minus_stft(key, subject_array, subject_array, num, get_middle_value)
                            high_num += 1
                    elif index['fatigue_level'] == 'low':  # good spirits
                        eeg_label.append(1)
                        if minus_stft_visualize:
                            num = 'low_' + str(low_num)
                            plot_minus_stft(key, subject_array, subject_array, num, get_middle_value)
                            low_num += 1

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


# train A, test B
def process_inter_stft_subjects_data(subjects_trials_data):
    eeg_data_output = {}
    eeg_label_output = {}
    for key, subjects_data in subjects_trials_data.items():
        eeg_data = []
        eeg_label = []
        for record_index, record_data in subjects_data.items():  # record0
            for index in record_data['trials_data']:
                if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                    subject_array = index['stft_spectrum']
                    eeg_data.append(subject_array)
                    #################################################################
                    if index['fatigue_level'] == 'high':  # tired
                        eeg_label.append(0)

                    elif index['fatigue_level'] == 'low':  # good spirits
                        eeg_label.append(1)

        new_dict = {key: eeg_data}
        eeg_data_output.update(new_dict)

        new_dict = {key: eeg_label}
        eeg_label_output.update(new_dict)

    return eeg_data_output, eeg_label_output


def train_stft_data(fft_eeg_data, fft_eeg_label, model_mode='cnn', minus_stft_mode=0):
    ################## 10-fold cross validation ###################
    # index = [i for i in range(len(fft_eeg_data))]
    # np.random.seed(0)
    # np.random.shuffle(index)
    # fft_eeg_data = fft_eeg_data[index]
    # fft_eeg_label = fft_eeg_label[index]
    ###############################################################

    fft_eeg_label = to_categorical(fft_eeg_label)

    if model_mode == 'cnn':
        input_shape = fft_eeg_data.shape[1:]
        model = call_cnn_model(input_shape)
        model.save_weights('init_model.hdf5')
    model.summary()
    if minus_stft_mode == 1:
        acc_avl_file = 'stft_rawdata_acc_loss'
        confusion_file = 'stft_rawdata_confusion'
        model_file = 'stft_rawdata.h5'
    elif minus_stft_mode == 2:
        acc_avl_file = 'stft_norm_acc_loss'
        confusion_file = 'stft_norm_confusion'
        model_file = 'stft_norm.h5'
    elif minus_stft_mode == 5:
        acc_avl_file = '10fold_mode5_stft_norm_acc_loss'
        confusion_file ='10fold_mode5_stft_norm_confusion'
        model_file = 'mode5_stft_norm.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=None, y_val=None, classes=2)

    acc = []
    loss = []
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    for train_index, test_index in cv.split(fft_eeg_data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = fft_eeg_data[train_index], fft_eeg_data[test_index]
        y_train, y_test = fft_eeg_label[train_index], fft_eeg_label[test_index]
        confusionMatrix.x_val = x_test
        confusionMatrix.y_val = y_test
        # confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=x_test, y_val=y_test, classes=2)
        my_callbacks = [EarlyStopping(monitor="val_loss", patience=30),
                        ModelCheckpoint(
                            filepath="./train_weight/" + model_file,
                            save_best_only=True, verbose=1),
                        customCallback,
                        confusionMatrix
                        ]
        fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                                verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
        acc.append(max(fittedModel.history["val_accuracy"]))
        loss.append(min(fittedModel.history["val_loss"]))
        # tf.compat.v1.reset_default_graph()
        K.clear_session()
        model.load_weights('init_model.hdf5')

    k_fold_cross = {"val_accuracy": np.array(acc), "val_loss": np.array(loss)}

    # model.save('fatigue_predict_stft_cnn.h5')
    return k_fold_cross


def train_inter_stft_data(id, x_train, y_train, x_test, y_test, model_mode='cnn', minus_stft_mode=0):
    ################## inter ###################
    index = [i for i in range(len(x_train))]
    np.random.seed(0)
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    index = [i for i in range(len(x_test))]
    np.random.seed(0)
    np.random.shuffle(index)
    x_test = x_test[index]
    y_test = y_test[index]
    ###############################################################

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if model_mode == 'cnn':
        input_shape = x_train.shape[1:]
        model = call_cnn_model(input_shape)
    model.summary()
    if minus_stft_mode == 1:
        acc_avl_file = id + '_stft_rawdata_acc_loss'
        confusion_file = id + '_stft_rawdata_confusion'
        model_file = 'stft_rawdata.h5'
    elif minus_stft_mode == 2:
        acc_avl_file = id + '_stft_norm_acc_loss'
        confusion_file = id + '_stft_norm_confusion'
        model_file = 'stft_norm.h5'

    elif minus_stft_mode == 5:
        acc_avl_file = id + '_mode5_stft_norm_acc_loss'
        confusion_file = id + '_mode5_stft_norm_confusion'
        model_file = '_mode5_stft_norm.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=x_test, y_val=y_test, classes=2)

    my_callbacks = [EarlyStopping(monitor="val_loss", patience=50),
                    ModelCheckpoint(filepath=("./train_weight/except_" + id + '_' + model_file),
                                    save_best_only=True, verbose=1),
                    customCallback,
                    confusionMatrix
                    ]
    fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                            verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
    acc = (max(fittedModel.history["val_accuracy"]))
    loss = (min(fittedModel.history["val_loss"]))

    return acc, loss


def butter_bandpass_filter_filtfilt(data, low_cut, high_cut, fs, order=5):
    wn1 = 2 * low_cut / fs
    wn2 = 2 * high_cut / fs
    [b, a] = butter(order, [wn1, wn2], btype="bandpass", output="ba")
    y = filtfilt(b, a, data)
    return y


if __name__ == '__main__':
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

    ################### parameter #####################
    feature_type = 'stft'  # 'stft' or 'time'
    data_normalize = False

    get_middle_value = False
    baseline_stft_visualize = False
    all_pepeole = False  # True: 10-fold , False:用A訓練 B測試
    minus_stft_visualize = False
    fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
    minus_stft_mode = 1  # 1: rawdata-baseline  2:(rawdata-baseline)normalize  5:(minus_data)normalize
    selected_channels = None

    loader = DatasetLoader()
    loader.apply_bandpass_filter = True
    loader.minus_mode = minus_stft_mode

    if data_normalize:
        loader.apply_signal_normalization = True
    else:
        loader.apply_signal_normalization = False

    if get_middle_value:
        loader.get_middle_value = True

    if all_pepeole:  # 10 fold
        subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="stft",
                                                                  # single_subject='c95ths',
                                                                  fatigue_basis=fatigue_basis,
                                                                  selected_channels=selected_channels
                                                                  )

        stft_eeg_data, stft_eeg_label = process_stft_subjects_data(subjects_trials_data, minus_stft_visualize)
        np.save("./npy_file/10fold_stft_eeg_data.npy", stft_eeg_data)
        np.save("./npy_file/10fold_stft_eeg_label.npy", stft_eeg_label)
        start_time = time.time()
        fittedModel = train_stft_data(stft_eeg_data, stft_eeg_label, model_mode='cnn', minus_stft_mode=minus_stft_mode)
        end_time = time.time()

        print('Training Time: ' + str(end_time - start_time))
        print('mean accuracy:%.3f' % fittedModel["val_accuracy"].mean())
        print(fittedModel["val_accuracy"].round(2))
        print('mean loss:%.3f' % fittedModel["val_loss"].mean())
        print(fittedModel["val_loss"].round(2))

    else:  # train A, test B
        subject_ids = loader.get_subject_ids()

        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                                   # single_subject=id,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        test_stft_eeg_data, test_stft_eeg_label = process_inter_stft_subjects_data(subjects_trials_data)

        all_acc = []
        all_loss = []
        for id in subject_ids:
            x_train = []
            y_train = []
            x_test = np.array(test_stft_eeg_data[id])
            y_test = np.array(test_stft_eeg_label[id])

            subject_ids_train = subject_ids.copy()
            subject_ids_train.remove(id)  # remove test subject
            for i in subject_ids_train:  # get training eeg data and label
                x_train.extend(np.array(test_stft_eeg_data[i]))
                y_train.extend(np.array(test_stft_eeg_label[i]))
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            acc, loss = train_inter_stft_data(id, x_train, y_train, x_test, y_test, model_mode='cnn', minus_stft_mode=minus_stft_mode)
            all_acc.append(acc)
            all_loss.append(loss)
        all_acc = np.array(all_acc)
        all_loss = np.array(all_loss)

        print('mean acc: ' + str(all_acc.mean().round(4)))
        print(all_acc.round(2))
        print('mean loss: ' + str(all_loss.mean().round(4)))
        print(all_loss.round(2))