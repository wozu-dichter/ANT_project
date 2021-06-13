import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import os
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, freq_band_selection
from freqency_train import call_cnn_model
from train_model import plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from train_model import ConfusionMatrix


def normalize(array, normalization_mode='min_max'):
    if normalization_mode == "min_max":
        array = array - np.min(array, axis=0)
        array = array / np.max(array, axis=0)
    elif normalization_mode == "z_score":
        array = array - np.mean(array, axis=0, dtype=np.float64)
        array = array / np.std(array, axis=0, dtype=np.float64)
    elif normalization_mode == "mean_norm":
        array = array - np.mean(array, axis=0, dtype=np.float64)
    else:
        return array
    return array


def plot_acc_and_loss(history, subject_id=None, save_piture=False):
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(subject_id + ':model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()
    plt.pause(0.01)
    if save_piture:
        plt.savefig("./train_weight/acc/" + subject_id + "_acc_loss.png")
    plt.clf()


def get_baseline_average_stft_eeg(data, subject_id, baseline_normalize='False'):
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

            ############stft compute###############
            f, t, zxx = stft(channel_data,
                             fs=sample_rate,
                             nperseg=stft_nperseg,
                             noverlap=int(stft_nperseg * stft_noverlap_ratio))
            i_index = []
            selected_time = []
            selected_zxx = []
            for i in range(t.shape[0]):  # get stft middle value
                if t[i] < (len(channel_data) / sample_rate - 0.5) and t[i] > 0.5:
                    selected_time.append(t[i])
                    selected_zxx.append(zxx[:, i])
                    i_index.append(i)
            t = np.array(selected_time)
            zxx = np.array(selected_zxx).T
            f, zxx = freq_band_selection(f, abs(zxx), min_freq=stft_min_freq, max_freq=stft_max_freq)
            # plot_eeg_stft(channel_data, t, f, zxx, eeg_channel[ith_channel],subject_id,num)
            # if baseline_normalize:
            #     zxx = normalize(zxx)
            stft_array.append(zxx)
            #####################################
        num = num + 1
        all_stft_array.append(np.array(stft_array))

    all_stft_array = np.array(all_stft_array).mean(axis=0)
    return t, f, all_stft_array, np.array(i_index)


def plot_eeg_stft(data, t, f, zxx, channel_label, subject_id, num):
    plt.subplot(211)
    plt.plot(data, label=channel_label)
    plt.title('raw data (time domain):' + channel_label)

    plt.subplot(212)
    plt.pcolormesh(t, f, np.abs(zxx), vmin=-2, vmax=10, shading='auto')
    plt.title('STFT Magnitude')
    plt.legend()
    plt.savefig("./train_weight/baseline/" + subject_id + "/" + subject_id + channel_label + '_' + str(num))
    # plt.waitforbuttonpress()
    plt.clf()


def plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num):
    for i in range(minus_subject_array.shape[2]):
        plt.subplot(311)
        plt.pcolormesh(t, f, np.abs(subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title(key + ':raw data in' + eeg_channel[i])
        plt.colorbar()
        plt.subplot(312)
        plt.pcolormesh(t, f, np.abs(baseline_eeg[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('baseline')
        plt.colorbar()
        plt.subplot(313)
        plt.pcolormesh(t, f, np.abs(minus_subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('after minus')
        plt.colorbar()

        plt.savefig("./train_weight/minus_stft_middle/" + key + "/" + key + '_' + eeg_channel[i] + '_' + num)
        plt.clf()


def process_subjects_data(subjects_trials_data, baseline_eeg_all, i_index, get_middle_value=True):
    eeg_data = []
    eeg_label = []

    high_num = 1
    low_num = 1
    for key, subjects_data in subjects_trials_data.items():
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                baseline_eeg = baseline_eeg_all[key].transpose((1, 2, 0))
                subject_array = index['stft_spectrum']
                if get_middle_value:
                    subject_array = subject_array[:, i_index, :]  # get stft middle value->[25,82]
                # for i in range(subject_array.shape[2]):  # normalize
                #     subject_array[:, :, i] = normalize(subject_array[:, :, i])
                #     baseline_eeg[:, :, i] = normalize(baseline_eeg[:, :, i])

                minus_subject_array = subject_array - baseline_eeg
                # for i in range(minus_subject_array.shape[2]):  #minus array normalize
                #     minus_subject_array[:, :, i] = normalize(minus_subject_array[:, :, i])

                if index['fatigue_level'] == 'high':
                    eeg_data.append(minus_subject_array)  # tired
                    eeg_label.append(0)
                    num = 'high_' + str(high_num)
                    # plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num)
                    high_num += 1
                elif index['fatigue_level'] == 'low':
                    eeg_data.append(minus_subject_array)  # good spirits
                    eeg_label.append(1)
                    num = 'low_' + str(low_num)
                    # plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num)
                    low_num += 1

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

dataset_dir = './dataset2/'
subject_dirs = glob_sorted(dataset_dir + "/*")

baseline_output = {}
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
        data = load_npy(npy_paths)
        t, f, all_stft_array, i_index = get_baseline_average_stft_eeg(data, subject_id,
                                                                      baseline_normalize='True')  # get baseline stft
        new_dict = {subject_id: all_stft_array}
        baseline_output.update(new_dict)

    # for ith_channel in range(all_stft_array.shape[0]):  # save average stft picture
    #     plt.pcolormesh(t, f, np.abs(all_stft_array[ith_channel, :, :]), vmin=-2, vmax=10, shading='auto')
    #     plt.title(eeg_channel[ith_channel])
    #     plt.savefig("./train_weight/baseline/"+subject_id+"/avg_"+subject_id+'_'+eeg_channel[ith_channel])
    #     plt.waitforbuttonpress()
    #     plt.clf()
loader = DatasetLoader()
subject_ids = loader.get_subject_ids()

# for subject_id in subject_ids:
# baseline_eeg = baseline_output[subject_id]
subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                           single_subject='c95ths',
                                           fatigue_basis="by_feedback",  # by_time
                                           )
eeg_data, eeg_label = process_subjects_data(subjects_trials_data, baseline_output, i_index,
                                            get_middle_value=True)  # minus data

eeg_label = to_categorical(eeg_label)

# model = call_cnn_model()
# model.summary()

x_train, x_test, y_train, y_test = train_test_split(eeg_data, eeg_label, test_size=0.1)

customCallback = plot_acc_val()
confusionMatrix = ConfusionMatrix(x_val=x_test, y_val=y_test, classes=2)
my_callbacks = [EarlyStopping(monitor="val_loss", patience=100),
                ModelCheckpoint(
                    filepath="./train_weight/middle_stft.h5",
                    save_best_only=True, verbose=1), customCallback, confusionMatrix]

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])
fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                        verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
# model.save('fatigue_predict_stft_cnn.h5')
print('best accuracy:%.3f' % max(fittedModel.history["val_accuracy"]))
print('best loss:%.3f' % min(fittedModel.history["val_loss"]))
# plot_acc_and_loss(fittedModel, subject_id='all_stft_cnn', save_piture=True)
a = 0
