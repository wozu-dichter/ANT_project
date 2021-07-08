from scipy.fftpack import fft
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from EEGModels import EEGNet, EEGNet_TINA_TEST, call_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from train_model import ConfusionMatrix
from freqency_train import plot_acc_val


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


def train_FFT_data(X, Y, model_mode='cnn'):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    chans, samples, kernels = x_train.shape[2], x_train.shape[1], 1

    if model_mode == 'eegnet':
        x_train = x_train.reshape((x_train.shape[0], chans, samples, kernels))
        x_test = x_test.reshape((x_test.shape[0], chans, samples, kernels))
        model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                       dropoutRate=0.25, kernLength=250, F1=16, D=30, F2=480,
                       dropoutType='Dropout')

    elif model_mode == 'cnn':
        x_train = x_train.reshape((x_train.shape[0], samples, 1, chans))  # [N data, 149,1,32]
        x_test = x_test.reshape((x_test.shape[0], samples, 1, chans))
        model = call_cnn_model()
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    customCallback = plot_acc_val()
    confusionMatrix = ConfusionMatrix(x_val=x_test, y_val=y_test, classes=2)
    my_callbacks = [EarlyStopping(monitor="val_loss", patience=100),
                    ModelCheckpoint(
                        filepath="./train_weight/norm_fft_norm_again.h5",
                        save_best_only=True, verbose=1), confusionMatrix, customCallback]
    fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                            verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
    # new_dict = {subject_id: fittedModel}
    # output.update(new_dict)
    model.save('fatigue_predict.h5')
    print('best accuracy:%.3f' % max(fittedModel.history["val_accuracy"]))
    print('best loss:%.3f' % min(fittedModel.history["val_loss"]))

    # plot_acc_and_loss(fittedModel, subject_id='all_FFT_cnn', save_piture=True)
    return fittedModel


def plot_dataset_fft(subject_array, baseline_array, minus_array, figure_name,num):
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]
    for i in range(minus_array.shape[1]):
        max_axis=np.max([subject_array[:, i], baseline_array[:, i], minus_array[:, i]])
        min_axis=np.min([subject_array[:, i], baseline_array[:, i], minus_array[:, i]])
        x=np.linspace(0, 30, minus_array.shape[0])
        plt.subplot(311)
        plt.plot(x,subject_array[:, i])
        plt.title(figure_name + ':raw data in' + eeg_channel[i])
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.subplot(312)
        plt.plot(x,baseline_array[:, i])
        plt.title('baseline')
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.subplot(313)
        plt.plot(x,minus_array[:, i])
        plt.title('after minus')
        plt.ylim(min_axis, max_axis)
        plt.tight_layout()
        plt.savefig("./train_weight/minus_fft/" + figure_name + '_' + eeg_channel[i])
        plt.clf()


def process_subjects_data(subjects_trials_data, baseline_output):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        baseline_array = baseline_output[key]
        high_num = 1
        low_num = 1
        sampling_rate = 512
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                subject_array = index['eeg'].T
                ################## FFT ####################
                fft_array = [(abs(fft(i)) / sampling_rate)[:len(abs(fft(i))) // 2] for i in subject_array]
                fft_array = np.array(fft_array).T
                subject_array = fft_array[1:int((high_fre / (sampling_rate // 2)) * len(fft_array)), :]
                ###### normalize ######

                # for i in range(subject_array.shape[1]):
                #     subject_array[:, i] = normalize(subject_array[:, i])
                #     baseline_array[:,i] = normalize(baseline_array[:,i])

                minus_array = subject_array - baseline_array

                # for i in range(minus_array.shape[1]):
                #     minus_array[:, i] = normalize(minus_array[:, i])

                if index['fatigue_level'] == 'high':
                    figure_name = key + "/" + key + "fft_high_" + str(high_num)
                    plot_dataset_fft(subject_array, baseline_array, minus_array, figure_name, high_num)
                    eeg_data.append(minus_array)  # tired
                    eeg_label.append(0)
                    high_num += 1

                elif index['fatigue_level'] == 'low':
                    figure_name = key + "/" + key + "fft_low_" + str(low_num)
                    plot_dataset_fft(subject_array, baseline_array, minus_array, figure_name, low_num)
                    eeg_data.append(minus_array)  # good spirits
                    eeg_label.append(1)
                    low_num += 1

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)
    # npy_name='./npy_file/rest/'+id+'_rest.npy'
    # np.save('./npy_file/rest/'+id+'_rest.npy',eeg_data) #[120,1280,32]
    # np.save('./npy_file/label/'+id+'_label.npy', eeg_label) #[120]
    return eeg_data, eeg_label


def get_baseline_FFT(data, subject_id):
    sampling_rate = 512
    all_fft_array = []
    high_fre = 30
    for i in range(0, 60, 5):
        raw_data = data['eeg'][i:i + 5, :, :].reshape(-1, 32)
        ############FFT compute###############
        fft_array = np.array([(abs(fft(i)) / sampling_rate)[:len(abs(fft(i))) // 2] for i in
                              raw_data.T]).T  # output shape = [point, channel]
        fft_array = fft_array[1:int((high_fre / (sampling_rate // 2)) * len(fft_array)), :]
        ################## normalize ###################

        ###############################################
        all_fft_array.append(np.array(fft_array))  # list: 12 array
    avg_stft_array = np.array(all_fft_array).mean(axis=0)  # 對12筆資料做平均
    # plot_dataset_fft(avg_stft_array,subject_id)
    # npy_name='./npy_file/baseline_fft/'+subject_id+'_baseline'
    # np.save(npy_name,avg_stft_array) #[1280,32]
    return avg_stft_array


# baseline之FFT獲取
dataset_dir = './dataset2/'
subject_dirs = glob_sorted(dataset_dir + "/*")
high_fre = 30
baseline_output = {}

for subject_dir in subject_dirs:  # get FFT baseline
    subject_id = os.path.basename(subject_dir)
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
        data = load_npy(npy_paths)
        avg_stft_array = get_baseline_FFT(data, subject_id)  # get baseline fft
        new_dict = {subject_id: avg_stft_array}
        baseline_output.update(new_dict)

all_pepeole = False
loader = DatasetLoader()
loader.apply_bandpass_filter = False

if all_pepeole:
    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                               fatigue_basis="by_feedback",
                                               # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                               )
    fft_eeg_data, fft_eeg_label = process_subjects_data(subjects_trials_data, baseline_output)
    np.save("./npy_file/fft_eeg_data.npy", fft_eeg_data)
    np.save("./npy_file/fft_eeg_label.npy", fft_eeg_label)
else:
    subject_ids = loader.get_subject_ids()
    for id in subject_ids:
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                                   single_subject=id,
                                                   fatigue_basis="by_feedback",
                                                   # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                   )
        fft_eeg_data, fft_eeg_label = process_subjects_data(subjects_trials_data, baseline_output)

# fittedModel = train_FFT_data(fft_eeg_data, fft_eeg_label)

a = 0
