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
import umap
from numpy import argmax
from scipy import signal


def plot_dataset_X(dataset_X, figure_name):
    rows, cols = dataset_X.shape

    plt.figure(figure_name, figsize=(20, 16))
    for i in range(cols):
        plt.subplot(cols, 1, i + 1)
        plt.title('Signal_' + str(i))
        plt.plot(dataset_X[:, i])
        plt.tight_layout()
    plt.savefig("./train_weight/" + figure_name + "_2.png")


def min_max(X):
    X = X - np.min(X, axis=0)
    min_max = X / np.max(X, axis=0)
    return min_max


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


def call_rnn_model():
    # inputs = Input(shape=(16, 2500))
    inputs = Input(shape=(2500, 16))  # (timesteps, channel) by default
    rnn1 = LSTM(units=128, activation="sigmoid", return_sequences=True, return_state=True, dropout=0.5)(inputs)
    # rnn1 = LSTM(units=128, activation="sigmoid", return_sequences=True, return_state=True, dropout=0.5)(rnn1)
    # rnn1 = LSTM(units=128, activation="sigmoid", return_sequences=True, return_state=True, dropout=0.5)(rnn1)
    rnn1 = LSTM(units=128, activation="sigmoid", return_sequences=False, return_state=False, dropout=0.5)(rnn1)
    dense1 = Dense(units=32, activation="relu")(rnn1)
    # dense2 = Dense(units=64, activation="relu")(dense1)
    output = Dense(units=2, activation="softmax")(dense1)
    rnn = Model(inputs=inputs, outputs=output)
    rnn.summary()
    return rnn


"""
loader = DatasetLoader()
subjects_trials_data = loader.load_data("time", "rest")

eeg_data, eeg_label = process_subjects_data(subjects_trials_data)
eeg_label = to_categorical(eeg_label)

np.save('eeg_data.npy', eeg_data)
np.save('eeg_label.npy', eeg_label)
"""

# eeg_data = np.load('eeg_data.npy')
# eeg_label = np.load('eeg_label.npy')

loader = DatasetLoader()
subject_ids = loader.get_subject_ids()
print(subject_ids)
output = {}

for subject_id in subject_ids:
    subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="time",
                                                              # single_subject=subject_id,
                                                              fatigue_basis="by_time",
                                                              # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                              )

    eeg_data, eeg_label = process_subjects_data(subjects_trials_data)
    eeg_label = to_categorical(eeg_label)

    x_train, y_train  = reformatted_data['train_x'][0], reformatted_data['train_y'][0]
    x_test, y_test = reformatted_data['valid_x'][0], reformatted_data['valid_y'][0]
    y_train, y_test =to_categorical(y_train), to_categorical(y_test)
    chans, samples, kernels = x_train.shape[2], x_train.shape[1], 1
    x_train = x_train.reshape((x_train.shape[0], chans, samples, kernels))
    x_test = x_test.reshape((x_test.shape[0], chans, samples, kernels))
    # model = call_rnn_model()
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
    print(subject_id)
    print('best accuracy:%.3f' % max(fittedModel.history["val_accuracy"]))
    print('best loss:%.3f' % min(fittedModel.history["val_loss"]))
    plt.figure(subject_id)
    plt.subplot(211)
    plt.plot(fittedModel.history['accuracy'])
    plt.plot(fittedModel.history['val_accuracy'])
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
    plt.savefig("./train_weight/acc/" + subject_id + "_acc_loss.png")
    plt.close()
    """ 
    layer_names = [layer.name for layer in model.layers]  # all layers name
    umap_data = np.concatenate((x_train, x_test))
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(name='middle_output').output)
    # eeg_data = eeg_data.reshape((umap_data.shape[0], chans, samples, kernels))

    intermediate_output = intermediate_layer_model.predict(umap_data)
    # eeg_label = argmax(y_test, axis=1)
    # intermediate_output = intermediate_layer_model.predict(x_test)

    eeg_len = intermediate_output.shape[0]
    reducer = umap.UMAP(n_neighbors=15, random_state=0)
    print("Umap")
    embedding = reducer.fit(intermediate_output.reshape(eeg_len, -1))
    embedding = embedding.embedding_

    embedding_train = embedding[:x_train.shape[0], :]
    embedding_test = embedding[x_train.shape[0]:, :]
    embedding_train_label = argmax(y_train, axis=1)
    embedding_test_label = argmax(y_test, axis=1)
    plt.scatter(embedding_train[np.where(embedding_train_label == 1), 0][0],
                embedding_train[np.where(embedding_train_label == 1), 1][0],
                c="r", s=5, label='train good')
    plt.scatter(embedding_train[np.where(embedding_train_label == 0), 0][0],
                embedding_train[np.where(embedding_train_label == 0), 1][0],
                c="b", s=5, label='train tired')
    plt.scatter(embedding_test[np.where(embedding_test_label == 1), 0][0],
                embedding_test[np.where(embedding_test_label == 1), 1][0],
                c="#d6b4fc", s=5, label='test good')
    print("test good num :{}".format(np.where(embedding_test_label == 1)[0].shape[0]))
    plt.scatter(embedding_test[np.where(embedding_test_label == 0), 0][0],
                embedding_test[np.where(embedding_test_label == 0), 1][0],
                c="#00ffff", s=5, label='test tired')
    print("test tired num :{}".format(np.where(embedding_test_label == 0)[0].shape[0]))

    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()
    plt.title(subject_id + ':UMAP EEG of rest', fontsize=24);
    plt.show()
    plt.savefig("./train_weight/umap/" + subject_id + "_Umap.png")
    """