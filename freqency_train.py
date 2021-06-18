from ANT_dataset_loader import DatasetLoader
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, Conv3D, MaxPool3D, GlobalAveragePooling2D, Conv2D, MaxPool2D, \
    BatchNormalization, GlobalAveragePooling3D, Dropout
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback


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


class plot_acc_val(Callback):
    # def on_epoch_end(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("End epoch {} of training; got log keys: {}".format(epoch, keys))
    def __init__(self, name='acc_loss'):
        super().__init__()
        self.name = name
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.count = 1
        plt.ion()
        print("init acc count: " + str(self.count))

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        self.plot_figure(epoch, logs)

    def on_train_end(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.count += 1
        plt.close(1)

    def plot_figure(self, epoch, logs):
        # self.logs.append(logs)
        self.losses.append(logs.get("loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_losses.append(logs.get("val_loss"))
        self.val_acc.append(logs.get("val_accuracy"))
        print("[Epoch{}]".format(epoch + 1))
        # Before plotting ensure at least 2 epochs have passed
        # if len(self.losses) > 1:
        n = np.arange(0, len(self.losses))
        # You can chose the style of your preference
        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.plot(n, self.acc, label="train_acc")
        plt.plot(n, self.val_acc, label="val_acc")
        plt.title("Training Accuracy [Epoch {}]".format(epoch + 1))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.legend()

        plt.subplot(212)
        plt.plot(n, self.losses, label="train_loss")
        plt.plot(n, self.val_losses, label="val_loss")
        plt.title("Training LOSS [Epoch {}]".format(epoch + 1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace "output" with whatever directory you want to put in the plots
        plt.show()
        plt.savefig("./train_weight/acc/" + self.name + "_" + str(self.count) + ".png")
        pass


def call_cnn_model(input_shape):
    # inputs = Input(shape=(25, 82, 32))
    inputs = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
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
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    cnn.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])
    return cnn


def fft_call_cnn_model(input_shape):
    # inputs = Input(shape=(25, 82, 32))
    inputs = Input(shape=(input_shape[0], 1, input_shape[1]))
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
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    cnn.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])
    return cnn


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
