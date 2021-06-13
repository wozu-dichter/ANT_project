import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, freq_band_selection
from freqency_train import call_cnn_model, normalize, plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from train_model import ConfusionMatrix
import time


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

        for ith_channel in range(raw_data.shape[1]):
            channel_data = raw_data[:, ith_channel]
            if data_normalize:
                channel_data = normalize(channel_data)
            ############stft compute###############
            f, t, zxx = stft(channel_data,
                             fs=sample_rate,
                             nperseg=stft_nperseg,
                             noverlap=int(stft_nperseg * stft_noverlap_ratio))
            i_index = []
            selected_time = []
            selected_zxx = []
            ################ get stft middle value #################
            if get_middle_value:
                for i in range(t.shape[0]):  # get stft middle value
                    if t[i] < (len(channel_data) / sample_rate - 0.5) and t[i] > 0.5:
                        selected_time.append(t[i])
                        selected_zxx.append(zxx[:, i])
                        i_index.append(i)
                t = np.array(selected_time)
                zxx = np.array(selected_zxx).T
            ##########################################
            f, zxx = freq_band_selection(f, abs(zxx), min_freq=stft_min_freq, max_freq=stft_max_freq)
            stft_array.append(zxx)
            #####################################
        num = num + 1
        all_stft_array.append(np.array(stft_array))

    all_stft_array = np.array(all_stft_array).mean(axis=0)
    return t, f, all_stft_array, np.array(i_index)


def plot_baseline_stft(all_stft_array, t, f, subject_id):
    for ith_channel in range(all_stft_array.shape[0]):  # save average stft picture
        plt.pcolormesh(t, f, np.abs(all_stft_array[ith_channel, :, :]), vmin=-2, vmax=10, shading='auto')
        plt.title(eeg_channel[ith_channel])
        file_name = "/avg_" + subject_id + '_' + eeg_channel[ith_channel]
        plt.savefig("./train_weight/baseline/" + subject_id + file_name)
        plt.waitforbuttonpress()
        plt.clf()


def plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num, get_middle_value):
    for i in range(minus_subject_array.shape[2]):
        plt.subplot(311)
        plt.pcolormesh(t, f, np.abs(subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
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


def process_stft_subjects_data(subjects_trials_data, baseline_data, minus_stft_visualize, minus_stft_mode, i_index,
                               get_middle_value=True):
    eeg_data = []
    eeg_label = []
    high_num = 1
    low_num = 1
    for key, subjects_data in subjects_trials_data.items():
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                subject_array = index['stft_spectrum']
                baseline_eeg = baseline_data[key].transpose((1, 2, 0))

                if get_middle_value:
                    subject_array = subject_array[:, i_index, :]  # get stft middle value:SHAPE->[25,82]
                #################### input normalize ###############################
                if minus_stft_mode == 0:  # data -baseline
                    minus_subject_array = subject_array - baseline_eeg
                elif minus_stft_mode == 1:  # (data -baseline) normalize
                    minus_subject_array = subject_array - baseline_eeg
                    for i in range(minus_subject_array.shape[2]):  # minus array normalize
                        minus_subject_array[:, :, i] = normalize(minus_subject_array[:, :, i])
                elif minus_stft_mode == 2:  # normalize data - normalize baseline
                    for i in range(subject_array.shape[2]):  # normalize
                        subject_array[:, :, i] = normalize(subject_array[:, :, i])
                        baseline_eeg[:, :, i] = normalize(baseline_eeg[:, :, i])
                    minus_subject_array = subject_array - baseline_eeg
                #################################################################
                if index['fatigue_level'] == 'high':
                    eeg_data.append(minus_subject_array)  # tired
                    eeg_label.append(0)
                    if minus_stft_visualize:
                        num = 'high_' + str(high_num)
                        plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num, get_middle_value)
                        high_num += 1
                elif index['fatigue_level'] == 'low':
                    eeg_data.append(minus_subject_array)  # good spirits
                    eeg_label.append(1)
                    if minus_stft_visualize:
                        num = 'low_' + str(low_num)
                        plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num, get_middle_value)
                        low_num += 1

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


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
    if minus_stft_mode == 0:
        acc_avl_file = 'stft_rawdata_acc_loss'
        confusion_file = 'stft_rawdata_confusion'
        model_file = 'stft_rawdata.h5'
    elif minus_stft_mode == 1:
        acc_avl_file = 'stft_norm_acc_loss'
        confusion_file = 'stft_norm_confusion'
        model_file = 'stft_norm.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=None, y_val=None, classes=2)

    acc=[]
    loss=[]
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    for train_index, test_index in cv.split(fft_eeg_data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = fft_eeg_data[train_index], fft_eeg_data[test_index]
        y_train, y_test = fft_eeg_label[train_index], fft_eeg_label[test_index]
        confusionMatrix.x_val=x_test
        confusionMatrix.y_val=y_test
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

    k_fold_cross = {"val_accuracy" : np.array(acc),"val_loss": np.array(loss)}

    # model.save('fatigue_predict_stft_cnn.h5')
    return k_fold_cross


eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

################### parameter #####################
data_normalize = False
get_middle_value = True
baseline_stft_visualize = False
all_pepeole = True
minus_stft_visualize = False
fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
minus_stft_mode = 0  # 0: rawdata-baseline  1:(rawdata-baseline)normalize

######################## get average baseline eeg ###########################
baseline_output = {}
for subject_dir in glob_sorted('./dataset2/*'):
    subject_id = os.path.basename(subject_dir)
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
        data = load_npy(npy_paths)
        t, f, all_stft_array, i_index = get_baseline_average_stft_eeg(data, subject_id, data_normalize=data_normalize,
                                                                      get_middle_value=get_middle_value)  # get baseline stft
        new_dict = {subject_id: all_stft_array}
        baseline_output.update(new_dict)
    if baseline_stft_visualize:
        plot_baseline_stft(all_stft_array, t, f, subject_id)
######################### load rest data ################################
loader = DatasetLoader()
if data_normalize:
    loader.apply_signal_normalization = True
else:
    loader.apply_signal_normalization = False

if all_pepeole:
    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                               fatigue_basis=fatigue_basis,
                                               # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                               )
    fft_eeg_data, fft_eeg_label = process_stft_subjects_data(subjects_trials_data, baseline_output,
                                                             minus_stft_visualize, minus_stft_mode, i_index,
                                                             get_middle_value=get_middle_value)
    np.save("./npy_file/fft_eeg_data.npy", fft_eeg_data)
    np.save("./npy_file/fft_eeg_label.npy", fft_eeg_label)
    start_time = time.time()
    fittedModel = train_stft_data(fft_eeg_data, fft_eeg_label, model_mode='cnn')
    end_time = _time = time.time()

else:
    subject_ids = loader.get_subject_ids()
    for id in subject_ids:
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                                   single_subject=id,
                                                   fatigue_basis=fatigue_basis,
                                                   # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                   )
        fft_eeg_data, fft_eeg_label = process_stft_subjects_data(subjects_trials_data, baseline_output,
                                                                 minus_stft_visualize, minus_stft_mode, i_index,
                                                                 get_middle_value=get_middle_value)
print('Training Time: ' + str(end_time - start_time))
print('mean accuracy:%.3f' % fittedModel["val_accuracy"].mean())
print('mean loss:%.3f' % fittedModel["val_loss"].mean())
