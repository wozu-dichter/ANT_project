import matplotlib.pyplot as plt
import numpy as np
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, signal_sticking
from freqency_train import fft_call_cnn_model, normalize, plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from train_model import ConfusionMatrix
import time
from scipy.signal import butter, filtfilt


def butter_bandpass_filter_filtfilt(data, low_cut, high_cut, fs, order=5):
    wn1 = 2 * low_cut / fs
    wn2 = 2 * high_cut / fs
    [b, a] = butter(order, [wn1, wn2], btype="bandpass", output="ba")
    y = filtfilt(b, a, data)
    return y


def plot_dataset_fft(subject_array, baseline_array, minus_array, figure_name):
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]
    for i in range(minus_array.shape[1]):
        max_axis = np.max([subject_array[:, i], baseline_array[:, i], minus_array[:, i]])
        min_axis = np.min([subject_array[:, i], baseline_array[:, i], minus_array[:, i]])
        x = np.linspace(0, 30, minus_array.shape[0])
        plt.subplot(311)
        plt.plot(x, subject_array[:, i])
        plt.title(figure_name + ':raw data in' + eeg_channel[i])
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.subplot(312)
        plt.plot(x, baseline_array[:, i])
        plt.title('baseline')
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.subplot(313)
        plt.plot(x, minus_array[:, i])
        plt.title('after minus')
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.savefig("./train_weight/minus_fft/" + figure_name + '_' + eeg_channel[i])
        plt.clf()


def fft_process_subjects_data(subjects_trials_data,feature_type, minus_fft_visualize=False):
    eeg_data = []
    eeg_label = []

    for key, subjects_data in subjects_trials_data.items():  # c95ccy
        high_num = 1
        low_num = 1
        for record_index, record_data in subjects_data.items():  # record0
            for index in record_data['trials_data']:
                if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                    if feature_type=='fft':
                        subject_array = index['fft_baseline_removed']
                    elif feature_type=='psd':
                        subject_array = index['psd_baseline_removed']
                    eeg_data.append(subject_array)
                    if index['fatigue_level'] == 'high':
                        if minus_fft_visualize:
                            figure_name = key + "/" + key + "fft_high_" + str(high_num)
                            plot_dataset_fft(subject_array, figure_name)
                            high_num += 1
                        eeg_label.append(0)
                    elif index['fatigue_level'] == 'low':
                        if minus_fft_visualize:
                            figure_name = key + "/" + key + "fft_low_" + str(low_num)
                            plot_dataset_fft(subject_array, figure_name)
                            low_num += 1
                        eeg_label.append(1)

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)
    # npy_name='./npy_file/rest/'+id+'_fft_rest.npy'
    # np.save('./npy_file/rest/'+id+'_rest.npy',eeg_data) #[720,149,32]
    # np.save('./npy_file/label/'+id+'_label.npy', eeg_label) #[720]
    return eeg_data, eeg_label


def train_fft_data(fft_eeg_data, fft_eeg_label, model_mode='cnn', minus_fft_mode=0):
    fft_eeg_label = to_categorical(fft_eeg_label)

    if model_mode == 'cnn':
        input_shape = fft_eeg_data.shape[1:]
        model = fft_call_cnn_model(input_shape)
        fft_eeg_data = fft_eeg_data.reshape(fft_eeg_data.shape[0], input_shape[0], 1, input_shape[1])
        model.save_weights('init_model.hdf5')
    model.summary()
    if minus_fft_mode == 1:
        acc_avl_file = 'fft_rawdata_acc_loss'
        confusion_file = 'fft_rawdata_confusion'
        model_file = 'fft_rawdata.h5'
    elif minus_fft_mode == 2:
        acc_avl_file = '_mode5_fft_norm_acc_loss'
        confusion_file = '_mode5_fft_norm_confusion'
        model_file = 'fft_norm.h5'

    elif minus_fft_mode == 5:
        acc_avl_file = '_mode5_fft_norm_acc_loss'
        confusion_file = '_mode5_fft_norm_confusion'
        model_file = 'fft_norm.h5'

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
        my_callbacks = [EarlyStopping(monitor="val_loss", patience=50),
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


def process_inter_fft_subjects_data(subjects_trials_data, feature_type):
    eeg_data_output = {}
    eeg_label_output = {}
    for key, subjects_data in subjects_trials_data.items():
        eeg_data = []
        eeg_label = []
        for record_index, record_data in subjects_data.items():  # record0
            for index in record_data['trials_data']:
                if index['fatigue_level'] != None:
                    if feature_type == 'fft':
                        subject_array = index['fft_baseline_removed'].T
                    elif feature_type == 'psd':
                        subject_array = index['psd_baseline_removed'].T

                    eeg_data.append(subject_array)
                    #################################################################
                    if index['fatigue_level'] == 'high':
                        eeg_label.append(0)  # tired
                    elif index['fatigue_level'] == 'low':  # good spirits
                        eeg_label.append(1)
        new_dict = {key: eeg_data}
        eeg_data_output.update(new_dict)

        new_dict = {key: eeg_label}
        eeg_label_output.update(new_dict)

    return eeg_data_output, eeg_label_output


def train_inter_fft_data(id, x_train, y_train, x_test, y_test, model_mode='cnn', minus_fft_mode=1):
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
        model = fft_call_cnn_model(input_shape)
        x_train = x_train.reshape(x_train.shape[0], input_shape[0], 1, input_shape[1])
        x_test = x_test.reshape(x_test.shape[0], input_shape[0], 1, input_shape[1])
    model.summary()
    if minus_fft_mode == 1:
        acc_avl_file = id + '_fft_rawdata_acc_loss'
        confusion_file = id + '_fft_rawdata_confusion'
        model_file = 'fft_rawdata.h5'
    elif minus_fft_mode == 2:
        acc_avl_file = id + '_fft_norm_acc_loss'
        confusion_file = id + '_fft_norm_confusion'
        model_file = 'fft_norm.h5'

    elif minus_fft_mode == 5:
        acc_avl_file = id + '_mode5_fft_norm_acc_loss'
        confusion_file = id + '_mode5_fft_norm_confusion'
        model_file = 'fft_norm.h5'

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


if __name__ == '__main__':
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

    ################### parameter #####################
    feature_type = 'fft' #fft or psd
    data_normalize = False
    get_middle_value = False
    baseline_stft_visualize = False
    all_pepeole = True  # True: 10-fold , False:用A訓練 B測試
    minus_fft_visualize = False
    fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
    minus_fft_mode = 1  # 1: rawdata-baseline  2:(rawdata-baseline)normalize
    selected_channels = None
    ######################### load rest data ################################
    loader = DatasetLoader()
    loader.apply_bandpass_filter = True
    loader.minus_mode = minus_fft_mode
    if data_normalize:
        loader.apply_signal_normalization = True
    else:
        loader.apply_signal_normalization = False

    if all_pepeole:  # 10 fold
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type=feature_type,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        stft_eeg_data, stft_eeg_label = fft_process_subjects_data(subjects_trials_data,feature_type, minus_fft_visualize)
        np.save("./npy_file/10fold_fft_eeg_data.npy", stft_eeg_data)
        np.save("./npy_file/10fold_fft_eeg_label.npy", stft_eeg_label)
        start_time = time.time()
        fittedModel = train_fft_data(stft_eeg_data, stft_eeg_label, model_mode='cnn', minus_fft_mode=minus_fft_mode)
        end_time = time.time()

        print('Training Time: ' + str(end_time - start_time))
        print('mean accuracy:%.3f' % fittedModel["val_accuracy"].mean())
        print(fittedModel["val_accuracy"].round(2))
        print('mean loss:%.3f' % fittedModel["val_loss"].mean())
        print(fittedModel["val_loss"].round(2))

    else:  # train A, test B
        subject_ids = loader.get_subject_ids()

        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type=feature_type,
                                                   # single_subject=id,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        test_fft_eeg_data, test_fft_eeg_label = process_inter_fft_subjects_data(subjects_trials_data, feature_type)

        all_acc = []
        all_loss = []
        for id in subject_ids:
            x_train = []
            y_train = []
            x_test = np.array(test_fft_eeg_data[id])
            y_test = np.array(test_fft_eeg_label[id])

            subject_ids_train = subject_ids.copy()
            subject_ids_train.remove(id)  # remove test subject
            for i in subject_ids_train:  # get training eeg data and label
                x_train.extend(np.array(test_fft_eeg_data[i]))
                y_train.extend(np.array(test_fft_eeg_label[i]))
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            acc, loss = train_inter_fft_data(id, x_train, y_train, x_test, y_test, model_mode='cnn',
                                             minus_fft_mode=minus_fft_mode)
            all_acc.append(acc)
            all_loss.append(loss)
        all_acc = np.array(all_acc)
        all_loss = np.array(all_loss)
        print('mean acc: ' + str(all_acc.mean().round(4)))
        print(all_acc.round(2))
        print('mean loss: ' + str(all_loss.mean().round(4)))
        print(all_loss.round(2))
