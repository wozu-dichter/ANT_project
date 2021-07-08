from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy
import numpy as np

from freqency_train import time_domain_call_cnn_model, time_call_cnn_model, plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from train_model import ConfusionMatrix
from tensorflow.keras import optimizers


def process_time_subjects_data(subjects_trials_data):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        for eeg_array in subjects_data:
            if eeg_array['fatigue_level'] == 'high' or eeg_array['fatigue_level'] == 'low':
                data = eeg_array['eeg']
                eeg_data.append(data)
                if eeg_array['fatigue_level'] == 'high':
                    eeg_label.append(0)
                elif eeg_array['fatigue_level'] == 'low':
                    eeg_label.append(1)
    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)
    return eeg_data, eeg_label


def process_inter_time_subjects_data(subjects_trials_data):
    eeg_data_output = {}
    eeg_label_output = {}
    for key, subjects_data in subjects_trials_data.items():
        eeg_data = []
        eeg_label = []
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                eeg_data.append(index['eeg'])
                if index['fatigue_level'] == 'high':
                    eeg_label.append(0)  # tired
                elif index['fatigue_level'] == 'low':
                    # good spirits
                    eeg_label.append(1)
        new_dict = {key: eeg_data}
        eeg_data_output.update(new_dict)

        new_dict = {key: eeg_label}
        eeg_label_output.update(new_dict)

    return eeg_data_output, eeg_label_output


def train_stft_data(time_eeg_data, time_eeg_label, model_mode='cnn', normalize_mode='mean_norm'):
    # 10-fold cross
    time_eeg_label = to_categorical(time_eeg_label)
    if model_mode == 'cnn':
        input_shape = time_eeg_data.shape[1:]
        model = time_domain_call_cnn_model(input_shape)

    elif model_mode == 'eegnet':
        from EEGModels import EEGNet, EEGNet_CNN_COMBINE
        input_shape = time_eeg_data.shape[1:]
        chans = input_shape[-1]
        samples = input_shape[0]
        model = EEGNet_CNN_COMBINE(nb_classes=2, Chans=chans, Samples=samples,
                       dropoutRate=0.25, kernLength=250, F1=16, D=30, F2=480,
                       dropoutType='Dropout')

    model.save_weights('init_model.hdf5')
    model.summary()

    if normalize_mode == None:
        acc_avl_file = 'time_domain_acc_loss_raw_data'
        confusion_file = 'time_domain_confusion_raw_data'
        model_file = 'time_domain_raw_data.h5'
    else:
        acc_avl_file = 'time_domain_acc_loss_' + normalize_mode
        confusion_file = 'time_domain_confusion_' + normalize_mode
        model_file = 'time_domain_' + normalize_mode + '.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=None, y_val=None, classes=2)

    acc = []
    loss = []
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    for train_index, test_index in cv.split(time_eeg_data):
        x_train, x_test = time_eeg_data[train_index], time_eeg_data[test_index]
        y_train, y_test = time_eeg_label[train_index], time_eeg_label[test_index]
        if model_mode == 'cnn':
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1, x_train.shape[2]))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1, x_test.shape[2]))
        elif model_mode == 'eegnet':
            x_train = x_train.reshape(x_train.shape[0], chans, samples, 1)
            x_test = x_test.reshape(x_test.shape[0], chans, samples, 1)

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


def train_inter_time_data(id, x_train, y_train, x_test, y_test, model_mode='eegnet', normalize_mode='mean_norm'):
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
        model = time_call_cnn_model(input_shape)
        x_train = x_train.reshape(x_train.shape[0], input_shape[0], 1, input_shape[1])
        x_test = x_test.reshape(x_test.shape[0], input_shape[0], 1, input_shape[1])

    elif model_mode == 'eegnet':
        from EEGModels import EEGNet
        input_shape = x_train.shape[1:]
        chans = input_shape[-1]
        samples = input_shape[0]
        model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                       dropoutRate=0.25, kernLength=250, F1=16, D=30, F2=480,
                       dropoutType='Dropout')
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        x_train = x_train.reshape(x_train.shape[0], chans, samples, 1)
        x_test = x_test.reshape(x_test.shape[0], chans, samples, 1)

    model.summary()

    if normalize_mode == None:
        acc_avl_file = 'eegnet_time_domain_acc_loss_raw_data_' + id
        confusion_file = 'eegnet_time_domain_confusion_raw_data_' + id
        model_file = 'eegnet_time_domain_raw_data.h5'
    else:
        acc_avl_file = 'eegnet_time_domain_acc_loss_' + normalize_mode + '_' + id
        confusion_file = 'eegnet_time_domain_confusion_' + normalize_mode + '_' + id
        model_file = 'eegnet_time_domain_' + normalize_mode + '.h5'

    customCallback = plot_acc_val(name=acc_avl_file)
    confusionMatrix = ConfusionMatrix(name=confusion_file, x_val=x_test, y_val=y_test, classes=2)

    my_callbacks = [EarlyStopping(monitor="val_loss", patience=200),
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


all_pepeole = False
loader = DatasetLoader()
loader.apply_signal_normalization = True
loader.apply_bandpass_filter = True


if loader.apply_signal_normalization == True:
    loader.normalization_mode = "mean_norm"
    normalization_mode = loader.normalization_mode
else:
    normalization_mode = None

if all_pepeole:
    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                               fatigue_basis="by_feedback",
                                               # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                               )

    time_eeg_data, time_eeg_label = process_time_subjects_data(subjects_trials_data)
    k_fold_cross = train_stft_data(time_eeg_data, time_eeg_label, model_mode='cnn')
    print(k_fold_cross)
    print('mean accuracy:%.3f' % k_fold_cross["val_accuracy"].mean())
    print('mean loss:%.3f' % k_fold_cross["val_loss"].mean())

else:
    subject_ids = loader.get_subject_ids()

    subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="time",
                                               # single_subject=id,
                                               fatigue_basis="by_feedback",
                                               # selected_channels=selected_channels
                                               )
    time_eeg_data, time_eeg_label = process_inter_time_subjects_data(subjects_trials_data)

    all_acc = []
    all_loss = []
    for id in subject_ids:
        print(id)
        x_train = []
        y_train = []
        x_test = np.array(time_eeg_data[id])
        y_test = np.array(time_eeg_label[id])

        subject_ids_train = subject_ids.copy()
        subject_ids_train.remove(id)  # remove test subject
        for i in subject_ids_train:  # get training eeg data and label
            x_train.extend(np.array(time_eeg_data[i]))
            y_train.extend(np.array(time_eeg_label[i]))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        acc, loss = train_inter_time_data(id, x_train, y_train, x_test, y_test, model_mode='eegnet',
                                          normalize_mode=normalization_mode)
        all_acc.append(acc)
        all_loss.append(loss)
    all_acc = np.array(all_acc)
    all_loss = np.array(all_loss)
    print(all_acc)
    print('mean acc: ' + str(all_acc.mean()))
    print('mean loss: ' + str(all_loss.mean()))
