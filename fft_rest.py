from scipy.fftpack import fft
from ANT_dataset_loader import DatasetLoader
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from EEGModels import EEGNet, EEGNet_TINA_TEST, call_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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
    plt.close()


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
        x_train = x_train.reshape((x_train.shape[0], samples, 1, chans))  # [N data, 256,1,32]
        x_test = x_test.reshape((x_test.shape[0], samples, 1, chans))
        model = call_cnn_model()
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    # customCallback = plot_acc_val()
    my_callbacks = [EarlyStopping(monitor="val_loss", patience=100),
                    ModelCheckpoint(
                        filepath="./train_weight/fatigure_predict.hdf5",
                        save_best_only=True, verbose=1)]
    fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                            verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
    # new_dict = {subject_id: fittedModel}
    # output.update(new_dict)
    model.save('fatigue_predict.h5')
    print('best accuracy:%.3f' % max(fittedModel.history["val_accuracy"]))
    print('best loss:%.3f' % min(fittedModel.history["val_loss"]))

    plot_acc_and_loss(fittedModel, subject_id='all_FFT_cnn', save_piture=True)
    return fittedModel


def plot_dataset_X(dataset_X, figure_name, sampling_rate=512):
    low_freq = 1
    high_freq = 30
    cols = dataset_X.shape[1]
    plt.figure(figure_name, figsize=(16, 16))
    eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
                   "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
                   "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
                   "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]
    for i in range(cols):
        xf = np.arange(len(dataset_X))[:sampling_rate // 2]
        yf = dataset_X

        plt.subplot(16, 2, i + 1)
        plt.plot(xf[low_freq:high_freq], yf[low_freq:high_freq, i])
        # plt.plot(abs(dataset_X[:, i]))
        plt.title(eeg_channel[i])
        plt.tight_layout()
    # plt.waitforbuttonpress()
    plt.ylabel('amp')
    plt.xlabel('Frequency [Hz]')
    plt.savefig("./train_weight/FFT/" + figure_name + ".png")
    plt.close()


def process_subjects_data(subjects_trials_data):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        print(key)
        high_num = 1
        low_num = 1
        sampling_rate = 512
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                subject_array = index['eeg'].T
                subject_array = np.array([(abs(fft(i))/sampling_rate)[:sampling_rate//2] for i in subject_array]).T
                subject_array = subject_array[:40,:]

                if index['fatigue_level'] == 'high':
                    # plot_dataset_X(subject_array, figure_name=key + "/" + key + "fft__high_" + str(high_num))
                    eeg_data.append(subject_array)  # tired
                    eeg_label.append(0)
                    high_num += 1
                elif index['fatigue_level'] == 'low':
                    # plot_dataset_X(subject_array, figure_name=key + "/" + key + "fft__low_" + str(low_num))
                    eeg_data.append(subject_array)  # good spirits
                    eeg_label.append(1)
                    low_num += 1
    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


all_pepeole = True
loader = DatasetLoader()
loader.apply_bandpass_filter = False

if all_pepeole:
    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                               fatigue_basis="by_feedback",
                                               # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                               )
    fft_eeg_data, fft_eeg_label = process_subjects_data(subjects_trials_data)
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
        fft_eeg_data, fft_eeg_label = process_subjects_data(subjects_trials_data)

fittedModel = train_FFT_data(fft_eeg_data, fft_eeg_label)

a = 0
