from ANT_dataset_loader import DatasetLoader
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, Conv3D, MaxPool3D, GlobalAveragePooling2D, Conv2D, MaxPool2D, \
    BatchNormalization, GlobalAveragePooling3D, Dropout
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from train_model import plot_acc_val
from tensorflow.keras import optimizers


def plot_dataset_X(dataset_X, figure_name):
    cols = dataset_X.shape[2]
    plt.figure(figure_name, figsize=(16, 16))
    for i in range(cols):
        # x = np.linspace(3, 28, 25)
        plt.subplot(cols, 1, i + 1)
        # plt.title('Signal_' + str(i))
        # plt.yticks(dataset_X[:, :, i], x)
        time = np.linspace(0, 5, 104)
        freq = np.linspace(3, 28, 25)
        plt.pcolormesh(time, freq, dataset_X[:, :, i], shading="auto", vmin=0, vmax=0.1)
        plt.title(["P3", "Pz", "P4", "Oz", "O1", 'O2'][i])
        plt.tight_layout()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig("./train_weight/stft/" + figure_name + ".png")
    plt.close()


def process_subjects_data(subjects_trials_data):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        print(key)
        low_num = 0
        high_num = 0
        for index in subjects_data:
            subject_array = index['stft_spectrum'][:, :, :]

            if index['fatigue_level'] == 'high':
                eeg_data.append(subject_array)  # tired
                eeg_label.append(0)
                stft_data = subject_array[:, :,
                            [10, 11, 12, 15, 30, 31]]  # C3, C4, P3, Pz, P4, Oz -> P3, Pz, P4, Oz ,O1,O2

                plot_dataset_X(stft_data, figure_name=key + "/" + key + "_high_" + str(high_num))
                high_num += 1

            elif index['fatigue_level'] == 'low':
                eeg_data.append(subject_array)  # good spirits
                eeg_label.append(1)
                stft_data = subject_array[:, :, [6, 8, 10, 11, 12, 15]]

                plot_dataset_X(stft_data, figure_name=key + "/" + key + "_low_" + str(low_num))
                low_num += 1

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


def call_cnn_model():
    inputs = Input(shape=(25, 82, 32))
    x = Conv2D(filters=128, kernel_size=5, activation="relu", padding="same")(inputs)
    # x = MaxPool2D(pool_size=(16, 16))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=5, activation="relu", padding="same")(x)
    # x = MaxPool2D(pool_size=(16, 16))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=5, activation="relu", padding="same")(x)
    # x = MaxPool2D(pool_size=(16, 16))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=5, activation="relu", padding="same")(x)
    # x = MaxPool2D(pool_size=(16, 16))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(units=2, activation="softmax")(x)

    # Define the model.
    cnn = Model(inputs, outputs, name="2Dcnn")
    return cnn

if __name__ == '__main__':
    loader = DatasetLoader()
    subjects_trials_data, _ = loader.load_data("rest", "stft")

    eeg_data, eeg_label = process_subjects_data(subjects_trials_data)
    eeg_label = to_categorical(eeg_label)

    np.save('eeg_data_freq.npy', eeg_data)
    np.save('eeg_label_freq.npy', eeg_label)

    # eeg_data = np.load('eeg_data_freq.npy')
    # eeg_label = np.load('eeg_label_freq.npy')

    model = call_cnn_model()
    model.summary()

    customCallback = plot_acc_val()
    my_callbacks = [EarlyStopping(monitor="val_loss", patience=100),
                    ModelCheckpoint(
                        filepath="./train_weight/weights.hdf5",
                        save_best_only=True, verbose=1), customCallback]
    x_train, x_test, y_train, y_test = train_test_split(eeg_data, eeg_label, test_size=0.1)
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                            verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)

    a = 0
