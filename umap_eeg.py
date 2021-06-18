import umap, os
import numpy as np
import matplotlib.pyplot as plt
from numpy import argmax
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy
from STFT_10_fold_cross import get_baseline_average_stft_eeg, process_stft_subjects_data, \
    process_inter_stft_subjects_data
from FFT_10_fold_cross import get_baseline_FFT, fft_process_subjects_data, process_inter_fft_subjects_data
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
feature_type = 'time'  # 'time' or 'stft'
fatigue_basis = 'by_feedback'  # 'by_time' or 'by_feedback'
selected_channels = None  # ["O1", "O2", "P3", "Pz", "P4", "Oz"] or None
excluded_subjects = None  # ['someone'] or None
minus_stft_mode = 0  # 0  1
get_middle_value = True
data_normalize = False
minus_stft_visualize = False
# FFT
minus_fft_mode = 0
#####################################################
loader = DatasetLoader()
if get_middle_value:
    loader.get_middle_value = True
# loader.validation_split=0.5
if data_normalize:
    loader.apply_signal_normalization = True
else:
    loader.apply_signal_normalization = False
subject_ids = loader.get_subject_ids()
#################################################


if feature_type == 'stft':
    baseline_output = {}
    for subject_dir in glob_sorted('./dataset2/*'):
        subject_id = os.path.basename(subject_dir)
        for record_dir in glob_sorted(subject_dir + "/*"):
            npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
            data = load_npy(npy_paths)
            t, f, all_stft_array = get_baseline_average_stft_eeg(data, subject_id,
                                                                 data_normalize=data_normalize,
                                                                 get_middle_value=get_middle_value)  # get baseline stft
            new_dict = {subject_id: all_stft_array}
            baseline_output.update(new_dict)

    if all_people:
        subjects_trials_data, reformatted_data = loader.load_data(data_type="rest", feature_type=feature_type,
                                                                  fatigue_basis=fatigue_basis,
                                                                  selected_channels=selected_channels,
                                                                  )
        stft_eeg_data, stft_eeg_label = process_stft_subjects_data(subjects_trials_data, baseline_output,
                                                                   minus_stft_visualize, minus_stft_mode)
        umap_data = stft_eeg_data
        umap_label = stft_eeg_label
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

        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                                   # single_subject=id,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        test_stft_eeg_data, test_stft_eeg_label = process_inter_stft_subjects_data(subjects_trials_data,
                                                                                   baseline_output, minus_stft_mode)
        for subject_id in subject_ids:
            print(subject_id)

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
            umap_data = np.array(test_stft_eeg_data[subject_id])
            umap_label = np.array(test_stft_eeg_label[subject_id])
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
            plt.savefig("./train_weight/umap/" + subject_id + '_mode_' + str(minus_stft_mode) + "_FFT_Umap.png")
            plt.close()

            # embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
            # plt.scatter(embedding.embedding_[:, 0], embedding.embedding_[:, 1], c=umap_label, cmap='Spectral', s=5)
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))
            # plt.title(subject_id + ':UMAP EEG of rest', fontsize=24);
            # plt.show()
            # plt.savefig("./train_weight/umap/" + subject_id + "_Umap.png")
            # plt.close()
elif feature_type == 'time':
    ######################## get average baseline eeg ###########################
    baseline_output = {}
    for subject_dir in glob_sorted('./dataset2/*'):
        subject_id = os.path.basename(subject_dir)
        for record_dir in glob_sorted(subject_dir + "/*"):
            npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
            data = load_npy(npy_paths)
            avg_stft_array = get_baseline_FFT(data, subject_id, data_normalize=data_normalize)  # get baseline stft
            new_dict = {subject_id: avg_stft_array}  # output shape = [149,32]
            baseline_output.update(new_dict)
    ######################### load rest data ################################
    if all_people:
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type=feature_type,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        stft_eeg_data, stft_eeg_label = fft_process_subjects_data(subjects_trials_data, baseline_output, minus_fft_mode)
        reducer = umap.UMAP(random_state=0)

        umap_data = stft_eeg_data
        umap_label = stft_eeg_label
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
        plt.savefig("./train_weight/umap/all_FFT_Umap.png")
        plt.close()

    else:
        print('sigle person')
        subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type=feature_type,
                                                   # single_subject=id,
                                                   fatigue_basis=fatigue_basis,
                                                   selected_channels=selected_channels
                                                   )
        test_fft_eeg_data, test_fft_eeg_label = process_inter_fft_subjects_data(subjects_trials_data,
                                                                                  baseline_output, minus_fft_mode)
        for subject_id in subject_ids:
            umap_data = np.array(test_fft_eeg_data[subject_id])
            umap_label = np.array(test_fft_eeg_label[subject_id])
            reducer = umap.UMAP(random_state=0)
            embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
            embedding = embedding.embedding_

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
            plt.savefig("./train_weight/umap/" + subject_id + '_mode_' + str(minus_fft_mode) + "_FFT_Umap.png")
            plt.close()
