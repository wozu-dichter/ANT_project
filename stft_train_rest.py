import matplotlib.pyplot as plt
import numpy as np
from ANT_dataset_loader import DatasetLoader
from freqency_train import call_cnn_model, normalize, plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from train_model import ConfusionMatrix
import time


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


def train_stft_data(reformatted_data, model_mode='cnn', minus_stft_mode=1, id=''):
    x_train, x_test, y_train, y_test = reformatted_data['train_x'], reformatted_data['valid_x'], reformatted_data[
        'train_y'], reformatted_data['valid_y']
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    if model_mode == 'cnn':
        input_shape = x_train.shape[1:]
        model = call_cnn_model(input_shape)
    model.summary()
    if minus_stft_mode == 1:
        acc_avl_file = id + '_stft_rawdata_acc_loss'
        confusion_file = id + '_stft_rawdata_confusion'
        model_file = id + '_stft_rawdata.h5'
    elif minus_stft_mode == 2:
        acc_avl_file = id + '_stft_norm_acc_loss'
        confusion_file = id + '_stft_norm_confusion'
        model_file = id + '_stft_norm.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=x_test, y_val=y_test, classes=2)
    my_callbacks = [EarlyStopping(monitor="val_loss", patience=50),
                    ModelCheckpoint(
                        filepath="./train_weight/" + model_file,
                        save_best_only=True, verbose=1), customCallback, confusionMatrix]

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                            verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
    # model.save('fatigue_predict_stft_cnn.h5')

    return fittedModel


eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

################### parameter #####################
data_normalize = False
get_middle_value = True
baseline_stft_visualize = False
all_pepeole = False
minus_stft_visualize = False
fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
minus_stft_mode = 1  # 1: rawdata-baseline  2:(rawdata-baseline)normalize

loader = DatasetLoader()
loader.minus_mode = minus_stft_mode
if data_normalize:
    loader.apply_signal_normalization = True
else:
    loader.apply_signal_normalization = False

if all_pepeole:
    subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="stft",
                                                              fatigue_basis=fatigue_basis,
                                                              # single_subject="c95ths",
                                                              # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                              )
    # np.save("./npy_file/fft_eeg_data.npy", fft_eeg_data)
    # np.save("./npy_file/fft_eeg_label.npy", fft_eeg_label)
    start_time = time.time()
    fittedModel = train_stft_data(reformatted_data, model_mode='cnn')
    end_time = _time = time.time()
    print('Training Time: ' + str(end_time - start_time))
    print('best accuracy:%.3f' % max(fittedModel.history["val_accuracy"]))
    print('best loss:%.3f' % min(fittedModel.history["val_loss"]))

else:
    subject_ids = loader.get_subject_ids()
    acc=[]
    loss=[]
    for id in subject_ids:
        subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="stft",
                                                                  single_subject=id,
                                                                  fatigue_basis=fatigue_basis,
                                                                  # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                                  )
        start_time = time.time()
        fittedModel = train_stft_data(reformatted_data, minus_stft_mode=minus_stft_mode,model_mode='cnn', id=id)
        end_time = _time = time.time()
        acc.append(max(fittedModel.history["val_accuracy"]).round(2))
        loss.append(min(fittedModel.history["val_loss"]).round(2))

    print('Training Time: ' + str(end_time - start_time))
    print('mean accuracy:%.3f' % np.mean(np.array(acc)))
    print(acc)
    print('mean loss:%.3f' % np.mean(np.array(loss)))
    print(loss)


