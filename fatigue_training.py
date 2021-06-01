from ANT_dataset_loader import DatasetLoader
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from EEGModels import EEGNet, EEGNet_TINA_TEST
from train_model import plot_acc_val
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt


def process_subjects_data(subjects_trials_data):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        print(key)
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                subject_array = index['eeg'].T
                if index['fatigue_level'] == 'high':
                    eeg_data.append(subject_array)  # tired
                    eeg_label.append(0)
                elif index['fatigue_level'] == 'low':
                    eeg_data.append(subject_array)  # good spirits
                    eeg_label.append(1)
    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


def plot_acc_and_loss(history, subject_id=None, save_piture=False):
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(subject_id + ':model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')

    plt.subplot(212)
    plt.plot(fittedModel.history['loss'])
    plt.plot(fittedModel.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()
    plt.pause(0.01)
    if save_piture:
        plt.savefig("./train_weight/acc/" + subject_id + "_acc_loss.png")
    plt.close()

loader = DatasetLoader()
subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="time",
                                                          # single_subject="c95ccy",
                                                          fatigue_basis="by_feedback",
                                                          # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                          )

# eeg_data, eeg_label = process_subjects_data(subjects_trials_data)
# eeg_label = to_categorical(eeg_label)

x_train, y_train = reformatted_data['train_x'], reformatted_data['train_y'] # shuffle
x_test, y_test = reformatted_data['valid_x'], reformatted_data['valid_y']

y_train, y_test = to_categorical(y_train), to_categorical(y_test)
chans, samples, kernels = x_train.shape[2], x_train.shape[1], 1
x_train = x_train.reshape((x_train.shape[0], chans, samples, kernels))
x_test = x_test.reshape((x_test.shape[0], chans, samples, kernels))

model = EEGNet_TINA_TEST(nb_classes=2, Chans=chans, Samples=samples,
                         dropoutRate=0.25, kernLength=250, F1=16, D=30, F2=480,
                         dropoutType='Dropout')
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

plot_acc_and_loss(fittedModel, subject_id='all', save_piture=True)

a=0