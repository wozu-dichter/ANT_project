import umap
import numpy as np
import matplotlib.pyplot as plt
from numpy import argmax
from ANT_dataset_loader import DatasetLoader
from EEGModels import EEGNet, EEGNet_TINA_TEST


def process_subjects_data(subjects_trials_data):
    eeg_data = []
    eeg_label = []
    for key, subjects_data in subjects_trials_data.items():
        for index in subjects_data:
            subject_array = index['eeg']
            if index['fatigue_level'] == 'high':
                eeg_data.append(subject_array)  # tired
                eeg_label.append(0)
            elif index['fatigue_level'] == 'low':
                eeg_data.append(subject_array)  # good spirits
                eeg_label.append(1)
    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label


################ parameter #####################################
all_people = True
feature_type = 'stft'  # 'time' or 'stft'
fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
selected_channels = None  # ["O1", "O2", "P3", "Pz", "P4", "Oz"] or None
excluded_subjects = None  # ['someone'] or None
#####################################################
loader = DatasetLoader()
loader.get_middle_value = True
# loader.validation_split=0.5

if all_people:
    subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type=feature_type,
                                                              fatigue_basis=fatigue_basis,
                                                              selected_channels=selected_channels,
                                                              )
    x_train, y_train = reformatted_data['train_x'], reformatted_data['train_y']
    x_test, y_test = reformatted_data['valid_x'], reformatted_data['valid_y']
    # y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    umap_data = np.concatenate((x_train, x_test))
    umap_label = np.concatenate((y_train, y_test))
    reducer = umap.UMAP(random_state=0)
    embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
    embedding = embedding.embedding_

    # embedding_train = embedding[:x_train.shape[0], :]
    # embedding_test = embedding[x_train.shape[0]:, :]

    plt.scatter(embedding[np.where(umap_label == 1), 0][0],
                embedding[np.where(umap_label == 1), 1][0],
                c="r", s=5, label='train tired')
    plt.scatter(embedding[np.where(umap_label == 0), 0][0],
                embedding[np.where(umap_label == 0), 1][0],
                c="b", s=5, label='train good')

    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()
    plt.title('all people :UMAP EEG of rest', fontsize=24);
    plt.show()
    plt.savefig("./train_weight/umap/all_don't_train_Umap.png")
    plt.close()

else:
    subject_ids = loader.get_subject_ids()
    for subject_id in subject_ids:
        print(subject_id)
        subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type=feature_type,
                                                                  excluded_subjects=[subject_id],
                                                                  fatigue_basis=fatigue_basis,
                                                                  selected_channels=selected_channels
                                                                  )
        # eeg_data, eeg_label = process_subjects_data(subjects_trials_data)
        # eeg_label = to_categorical(eeg_label)
        x_train, y_train = reformatted_data['train_x'], reformatted_data['train_y']
        x_test, y_test = reformatted_data['valid_x'], reformatted_data['valid_y']
        # y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        """
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
    
        fittedModel = model.fit(x_train, y_train, batch_size=200, epochs=200,
                                verbose=1, validation_data=(x_test, y_test), callbacks=my_callbacks)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(name="conv_4").output)
        intermediate_output = intermediate_layer_model.predict(eeg_data)
        
        eeg_len = intermediate_output.shape[0]
        """
        umap_data = np.concatenate((x_train, x_test))
        umap_label = np.concatenate((y_train, y_test))
        reducer = umap.UMAP(random_state=0)
        embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
        embedding = embedding.embedding_

        # embedding_train = embedding[:x_train.shape[0], :]
        # embedding_test = embedding[x_train.shape[0]:, :]

        plt.scatter(embedding[np.where(umap_label == 1), 0][0],
                    embedding[np.where(umap_label == 1), 1][0],
                    c="r", s=5, label='train tired')
        plt.scatter(embedding[np.where(umap_label == 0), 0][0],
                    embedding[np.where(umap_label == 0), 1][0],
                    c="b", s=5, label='train good')

        plt.gca().set_aspect('equal', 'datalim')
        plt.legend()
        plt.title(subject_id + ':UMAP EEG of rest', fontsize=24);
        plt.show()
        plt.savefig("./train_weight/umap/" + subject_id + "_don't_train_Umap.png")
        plt.close()

        # embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
        # plt.scatter(embedding.embedding_[:, 0], embedding.embedding_[:, 1], c=umap_label, cmap='Spectral', s=5)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))
        # plt.title(subject_id + ':UMAP EEG of rest', fontsize=24);
        # plt.show()
        # plt.savefig("./train_weight/umap/" + subject_id + "_Umap.png")
        # plt.close()
