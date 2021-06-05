import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import os
from ANT_dataset_loader import DatasetLoader,glob_sorted,load_npy,freq_band_selection
from freqency_train import call_cnn_model
from train_model import plot_acc_val
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers


def get_baseline_average_stft_eeg(data, subject_id):
    ############ get baselinestft eeg(average) #############
    stft_nperseg = 500
    stft_noverlap_ratio = 0.95
    stft_min_freq = 3
    stft_max_freq = 28
    sample_rate = 512

    all_stft_array = []
    num=0
    for i in range(0, 60, 5):
        stft_array = []
        raw_data = data['eeg'][i:i + 5, :, :].reshape(-1, 32)
        for ith_channel in range(raw_data.shape[1]):
            channel_data = raw_data[:, ith_channel]

            ############stft compute###############
            f, t, zxx = stft(channel_data,
                             fs=sample_rate,
                             nperseg=stft_nperseg,
                             noverlap=int(stft_nperseg * stft_noverlap_ratio))
            f, zxx = freq_band_selection(f, zxx, min_freq=stft_min_freq, max_freq=stft_max_freq)
            # plot_eeg_stft(channel_data, t, f, zxx, eeg_channel[ith_channel],subject_id,num)
            stft_array.append(zxx)
            #####################################
        num=num+1
        all_stft_array.append(np.array(stft_array))

    all_stft_array = np.array(all_stft_array).mean(axis=0)
    return t, f, all_stft_array


def plot_eeg_stft(data, t, f, zxx, channel_label,subject_id,num):
    plt.subplot(211)
    plt.plot(data, label=channel_label)
    plt.title('raw data (time domain):' + channel_label)

    plt.subplot(212)
    plt.pcolormesh(t, f, np.abs(zxx), vmin=-2, vmax=10, shading='auto')
    plt.title('STFT Magnitude')
    plt.legend()
    plt.savefig("./train_weight/baseline/"+subject_id+"/"+subject_id+channel_label+'_'+str(num))
    # plt.waitforbuttonpress()
    plt.clf()

def plot_minus_stft(key,minus_subject_array,subject_array,baseline_eeg,num):
    for i in range(minus_subject_array.shape[2]):
        plt.subplot(311)
        plt.pcolormesh(t, f, np.abs(subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title(key + ':raw data in' + eeg_channel[i])
        plt.colorbar()
        plt.subplot(312)
        plt.pcolormesh(t, f, np.abs(baseline_eeg[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('baseline')
        plt.colorbar()
        plt.subplot(313)
        plt.pcolormesh(t, f, np.abs(minus_subject_array[:, :, i]), vmin=-10, vmax=100, shading='auto')
        plt.title('after minus')
        plt.colorbar()

        plt.savefig("./train_weight/minus_stft/" + key + "/" + key + '_' + eeg_channel[i] + '_' + num)
        plt.close()

def process_subjects_data(subjects_trials_data, baseline_eeg):
    eeg_data = []
    eeg_label = []
    baseline_eeg = baseline_eeg.transpose((1,2,0))
    high_num=0
    low_num=0
    for key, subjects_data in subjects_trials_data.items():
        for index in subjects_data:
            if index['fatigue_level'] == 'high' or index['fatigue_level'] == 'low':
                high_num+=1
                low_num+=1
                subject_array = index['stft_spectrum']
                minus_subject_array = subject_array - baseline_eeg
                if index['fatigue_level'] == 'high':
                    eeg_data.append(minus_subject_array)  # tired
                    eeg_label.append(0)
                    num = 'high_' + str(high_num)
                    plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num)
                elif index['fatigue_level'] == 'low':
                    eeg_data.append(minus_subject_array)  # good spirits
                    eeg_label.append(1)
                    num='low_'+str(low_num)
                    plot_minus_stft(key, minus_subject_array, subject_array, baseline_eeg, num)

    eeg_data = np.array(eeg_data)
    eeg_label = np.array(eeg_label)

    return eeg_data, eeg_label

eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

dataset_dir = './dataset2/'
subject_dirs = glob_sorted(dataset_dir + "/*")

baseline_output={}
for subject_dir in subject_dirs:
    subject_id = os.path.basename(subject_dir)
    for record_dir in glob_sorted(subject_dir + "/*"):
        npy_paths = [p for p in glob_sorted(record_dir + "/*.npy") if 'baseline' in p][0]
        data = load_npy(npy_paths)
        t, f, all_stft_array = get_baseline_average_stft_eeg(data,subject_id)  #get baseline stft
        new_dict = {subject_id: all_stft_array}
        baseline_output.update(new_dict)

    # for ith_channel in range(all_stft_array.shape[0]):  # save average stft picture
    #     plt.pcolormesh(t, f, np.abs(all_stft_array[ith_channel, :, :]), vmin=-2, vmax=10, shading='auto')
    #     plt.title(eeg_channel[ith_channel])
    #     plt.savefig("./train_weight/baseline/"+subject_id+"/avg_"+subject_id+'_'+eeg_channel[ith_channel])
    #     plt.waitforbuttonpress()
    #     plt.clf()
loader = DatasetLoader()
subject_ids = loader.get_subject_ids()

# for subject_id in subject_ids:
baseline_eeg = baseline_output[subject_id]
subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type="stft",
                                                          # single_subject=subject_id,
                                                          fatigue_basis="by_feedback",   #by_time
                                                          )
eeg_data, eeg_label =process_subjects_data(subjects_trials_data, baseline_eeg) #minus data
eeg_label = to_categorical(eeg_label)

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
a=0
