import numpy as np
import os, copy, umap
from ANT_dataset_loader import glob_sorted, load_npy, freq_band_selection
import collections
from scipy.fftpack import fft
from freqency_train import normalize
import matplotlib.pyplot as plt
from scipy.signal import stft


def get_unity_data(time_range_ratio=0, path='./gaming_data/*'):
    eeg_data_output = {}
    for subject_dir in glob_sorted(path):
        subject_id = os.path.basename(subject_dir)
        count = 1
        subject_all_output = {}
        for record_dir in glob_sorted(subject_dir + "/*"):
            subject_output = {}
            raw_data = load_npy(record_dir)

            eeg_data = raw_data['eeg'][:-1]
            stageRecord = raw_data['stageRecord'][:-1]
            print(collections.Counter(stageRecord))
            unique, counts = np.unique(stageRecord, return_counts=True)
            subject_index = 0
            for i in range(len(unique)):
                subject_array = eeg_data[subject_index:subject_index + counts[i]]
                data_len = len(subject_array)
                time_start = int(data_len * time_range_ratio)
                print('time_start: {}'.format(time_start))
                subject_array = subject_array[time_start:]
                subject_index = subject_index + counts[i]
                subject_output.update({i: subject_array})

            subject_all_output.update({count: subject_output})
            count += 1
        eeg_data_output.update({subject_id: subject_all_output})
    return eeg_data_output


def get_fft_data(eeg_data_output, segment_time=5, sampling_rate=512, high_fre=30):
    fft_eeg_output = copy.deepcopy(eeg_data_output)
    t = segment_time
    for key, data in eeg_data_output.items():
        for index_data, raw_data in data.items():
            for count, raw_data_index in raw_data.items():  # raw_data in stage 0-5
                all_fft_array = []
                for i in range(0, len(raw_data_index), t):
                    raw_data_index = np.array(raw_data_index)
                    output = raw_data_index[i:i + t, :, :].reshape(-1, 32)
                    if raw_data_index[i:i + t, :, :].shape[0] == t:
                        ############FFT compute###############
                        fft_array = np.array([(abs(fft(i_fft)) / sampling_rate)[:len(abs(fft(i_fft))) // 2] for i_fft in
                                              output.T]).T  # output shape = [point, channel]
                        fft_array = fft_array[1:int((high_fre / (sampling_rate // 2)) * len(fft_array)), :]
                        ##############################################
                        all_fft_array.append(np.array(fft_array))  # list: N fft  array
                if count == 0:  # baseline
                    avg_stft_array = np.array(all_fft_array).mean(axis=0)  # 對N筆資料做平均
                    fft_eeg_output[key][index_data][count] = avg_stft_array
                else:
                    fft_eeg_output[key][index_data][count] = all_fft_array

    return fft_eeg_output


def minus_fft_data(fft_eeg_output, minus_mode=0):
    minus_fft_output = copy.deepcopy(fft_eeg_output)
    for key, array in fft_eeg_output.items():
        for i, stage in array.items():
            for stage_index, stage_array in stage.items():
                if stage_index == 0:
                    baseline = stage_array
                else:
                    for array_raw_data_i, array_raw_data in enumerate(stage_array):
                        if minus_mode == 0:
                            minus_array = array_raw_data - baseline
                        elif minus_mode == 1:
                            minus_array = normalize(array_raw_data - baseline)
                        minus_fft_output[key][i][stage_index][
                            array_raw_data_i] = minus_array  # output shae = [149,32] (t=5)

    return minus_fft_output


def get_data_process(minus_fft_output, get_all_data_together=True):
    all_out_put = {}
    all_out_put_label = {}
    if get_all_data_together:
        for key, array in minus_fft_output.items():
            output_label = []
            output_array = []
            for i, stage in array.items():
                all_stage_key = list(stage.keys())
                dict_you_want = {your_key: stage[your_key] for your_key in all_stage_key[1:]}  # without baseline
                for i_stage, stage_array in dict_you_want.items():
                    for index_stage_array in stage_array:
                        output_array.append(index_stage_array)
                        output_label.append(i_stage)
            all_out_put.update({key: np.array(output_array)})
            all_out_put_label.update({key: np.array(output_label)})
    else:
        for key, array in minus_fft_output.items():
            subject_array = {}
            subject_label = {}
            for i, stage in array.items():
                all_stage_key = list(stage.keys())
                dict_you_want = {your_key: stage[your_key] for your_key in all_stage_key[1:]}  # without baseline
                output_label = []
                output_array = []
                for i_stage, stage_array in dict_you_want.items():
                    for index_stage_array in stage_array:
                        output_array.append(index_stage_array)
                        output_label.append(i_stage)
                subject_array.update({i: np.array(output_array)})
                subject_label.update({i: np.array(output_label)})
            all_out_put.update({key: subject_array})
            all_out_put_label.update({key: subject_label})

    return all_out_put, all_out_put_label


def get_stft_data(eeg_data_output, segment_time=5, get_middle_value=False):
    sample_rate = 512
    stft_nperseg = 500
    stft_noverlap_ratio = 0.95
    stft_min_freq = 3
    stft_max_freq = 28
    fft_eeg_output = copy.deepcopy(eeg_data_output)
    # t = segment_time
    for key, data in eeg_data_output.items():
        for index_data, raw_data in data.items():
            for count, raw_data_index in raw_data.items():  # raw_data in stage 0-5
                all_fft_array = []
                for i_step in range(0, len(raw_data_index), segment_time):
                    stft_array = []
                    raw_data_index = np.array(raw_data_index)
                    output = raw_data_index[i_step:i_step + segment_time, :, :].reshape(-1, 32)
                    if raw_data_index[i_step:i_step + segment_time, :, :].shape[0] == segment_time:
                        ############ stft compute###############
                        for ith_channel in range(output.shape[1]):
                            channel_data = output[:, ith_channel]
                            f, t, zxx = stft(channel_data,
                                             fs=sample_rate,
                                             nperseg=stft_nperseg,
                                             noverlap=int(stft_nperseg * stft_noverlap_ratio))
                            selected_time = []
                            selected_zxx = []
                            ################ get stft middle value #################
                            if get_middle_value:
                                for i in range(t.shape[0]):  # get stft middle value
                                    if t[i] < (len(channel_data) / sample_rate - 0.5) and t[i] > 0.5:
                                        selected_time.append(t[i])
                                        selected_zxx.append(zxx[:, i])
                                t = np.array(selected_time)
                                zxx = np.array(selected_zxx).T
                            ##########################################
                            f, zxx = freq_band_selection(f, abs(zxx), min_freq=stft_min_freq, max_freq=stft_max_freq)
                            stft_array.append(zxx)
                        ##############################################
                        all_fft_array.append(np.array(stft_array).transpose((1, 2, 0)))  # list: N fft  array
                if count == 0:  # baseline
                    avg_stft_array = np.array(all_fft_array).mean(axis=0)  # 對N筆資料做平均
                    fft_eeg_output[key][index_data][count] = avg_stft_array
                else:
                    fft_eeg_output[key][index_data][count] = all_fft_array

    return fft_eeg_output


##############parameter#################3
get_all_data_together = False
################################
eeg_data_output = get_unity_data(time_range_ratio=0, path='./gaming_data/*')
fft_eeg_output = get_fft_data(eeg_data_output, segment_time=1)
# stft_eeg_output = get_stft_data(eeg_data_output, segment_time=1,get_middle_value=True)
minus_fft_output = minus_fft_data(fft_eeg_output, minus_mode=0)
all_out_put, all_out_put_label = get_data_process(minus_fft_output, get_all_data_together=get_all_data_together)

############## Umap ################
color = ['r', 'g', 'b', 'k', 'c']
for key, data in all_out_put.items():
    if get_all_data_together:
        umap_data = data
        umap_label = all_out_put_label[key]
        print(collections.Counter(umap_label))
        reducer = umap.UMAP(random_state=0)
        embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
        embedding = embedding.embedding_

        plt.figure()
        # plt.gca().set_aspect('equal', 'datalim')
        for i in range(1, len(collections.Counter(umap_label))+1, 1):
            print(i)
            plt.scatter(embedding[np.where(umap_label == i), 0][0],
                        embedding[np.where(umap_label == i), 1][0],
                        c=color[i - 1], s=10, label='stage ' + str(i))
        plt.legend()
        plt.title(key + ' :UMAP EEG of playing Unity', fontsize=24)
        fig_name = "./train_weight/unity/" + key + "_stft_Umap.png"
        plt.savefig(fig_name)
        plt.clf()
    else:
        for index, array in data.items():
            umap_data = array
            umap_label = all_out_put_label[key][index]
            reducer = umap.UMAP(random_state=0)
            embedding = reducer.fit(umap_data.reshape(umap_data.shape[0], -1))
            embedding = embedding.embedding_

            plt.figure()
            # plt.gca().set_aspect('equal', 'datalim')
            for i in range(1, len(collections.Counter(umap_label))+1, 1):
                plt.scatter(embedding[np.where(umap_label == i), 0][0],
                            embedding[np.where(umap_label == i), 1][0],
                            c=color[i - 1], s=10, label='stage ' + str(i))
            plt.legend()
            plt.title(key + ' :UMAP EEG of playing Unity', fontsize=24)
            fig_name = "./train_weight/unity/" + key + '_' + str(index) + "_stft__Umap.png"
            plt.savefig(fig_name)
            plt.clf()

a = 0
