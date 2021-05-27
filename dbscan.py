import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from ANT_dataset_loader import DatasetLoader
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras import layers


def Autoencoder():
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(2500, 16))
    Flatten = layers.Flatten()(input_img)

    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(512, activation='relu')(Flatten)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(2500 * 16, activation='sigmoid')(encoded)
    decoded = layers.Dense(1, activation='sigmoid')(decoded)
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    return autoencoder


def process_subjects_data(x_train, y_train):
    eeg_data = []
    eeg_label = []
    for data in x_train:
        for index in range(data.shape[0]):
            eeg_data.append(data[index, :, :])
    for index in range(len(y_train)):
        eeg_label.extend(y_train[index])
    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)
    return eeg_data, eeg_label


loader = DatasetLoader()
loader.apply_bandpass_filter = False
loader.normalization_mode = "min_max"
loader.validation_split = 0.1
subject_ids = loader.get_subject_ids()
subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type="time",
                                                          # single_subject=subject_id,
                                                          fatigue_basis="by_time",
                                                          # selected_channels=["C3", "C4", "P3", "Pz", "P4", "Oz"]
                                                          )
np.save('subjects_trials_data_6_bandpass_allpeople.npy', subjects_trials_data)
np.save('reformatted_data_6_bandpass_allpeople.npy', reformatted_data)
"""

reformatted_data = np.load('reformatted_data_6_bandpass_allpeople.npy',allow_pickle=True)
"""

x_train, y_train = reformatted_data['train_x'], reformatted_data['train_y']
valid_x, valid_y = reformatted_data['valid_x'], reformatted_data['valid_y']
eeg_data, eeg_label = process_subjects_data(x_train, y_train)
val_eeg_data, val_eeg_label = process_subjects_data(valid_x, valid_y)

model = Autoencoder()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
history = model.fit(
                    eeg_data,
                    eeg_label,
                    epochs=20,
                    batch_size=200,
                    )



plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
plt.pause(0.01)
plt.close()
a = 0

"""
# umap降維
import umap

reducer = umap.UMAP(random_state=0, n_components=2)
embedding = reducer.fit(eeg_data.reshape(eeg_data.shape[0], -1))
embedding = embedding.embedding_

# DBSCAN 聚類
# from sklearn.cluster import DBSCAN
#
# model = DBSCAN(eps=0.5, min_samples=5)
# model.fit(embedding)
# labels = model.fit_predict(embedding)
# plt.scatter(embedding[:, 0], embedding[:, 1],
#             c=labels, s=5)
# plt.title("DBSCAN")

# LDA聚類
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDA = LinearDiscriminantAnalysis(n_components=1)
lda_x = LDA.fit_transform(embedding, eeg_label)
plt.scatter(lda_x, lda_x,
            c=eeg_label, s=5)
plt.title("LDA")

"""
