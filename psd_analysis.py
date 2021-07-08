import os
import numpy as np
import matplotlib.pyplot as plt
import re
from custom_lib import glob_sorted
from scipy.signal import welch
from ANT_dataset_loader import DatasetLoader, glob_sorted, load_npy, freq_band_selection
from freqency_train import normalize

def create_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)

eeg_channel = ["Fp1", "Fp2", "F3", "Fz", "F4", "T7", "C3", "Cz",
               "C4", "T8", "P3", "Pz", "P4", "P7", "P8", "Oz",
               "AF3", "AF4", "F7", "F8", "FT7", "FC3", "FCz", "FC4",
               "FT8", "TP7", "CP3", "CPz", "CP4", "TP8", "O1", "O2"]

loader = DatasetLoader()
loader.apply_signal_normalization = False

subject_ids = loader.get_subject_ids()

subjects_trials_data, _ = loader.load_data(data_type="rest", feature_type='time',
                                           # single_subject=id,
                                           fatigue_basis='by_feedback'
                                           )
for key in subject_ids:
    eeg_subject = subjects_trials_data[key]
    subject_band_array_high = []
    subject_band_array_low = []
    subject_band_array_high_nor = []
    subject_band_array_low_nor = []
    baseline_path = './dataset2/' + key + '/1/*.npy'
    baseline_path = [p for p in glob_sorted(baseline_path) if 'baseline' in p][0]
    baseline_data = load_npy(baseline_path)
    for data in eeg_subject:
        if data['fatigue_level'] == 'high' or 'low':
            subject_array = data['eeg']
            band_array = []
            band_array_nor = []
            for i_channel in range(subject_array.shape[-1]):
                freqs, psd = welch(subject_array[:, i_channel], fs=512, nperseg=512)
                freqs_norm, psd_norm = welch(normalize(subject_array[:, i_channel]), fs=512, nperseg=512)

                psd = psd[(np.where((freqs > 0) & (freqs <= 30)))]
                psd_norm = psd_norm[(np.where((freqs > 0) & (freqs <= 30)))]
                n = np.arange(0, psd.shape[0])
                plt.clf()
                plt.subplot(211)
                plt.plot(n, psd)
                plt.title('raw data')
                plt.xlabel("PSD")
                plt.xlabel("freqency")
                plt.tight_layout()

                plt.subplot(212)
                plt.plot(n, psd_norm, label='norm')
                plt.title('normalize')
                plt.xlabel("PSD")
                plt.xlabel("freqency")
                plt.tight_layout()

                plt.suptitle(key + ':' + eeg_channel[i_channel])

                plt.show()
                plt.pause(0.05)
                plt.savefig('./psd_figures2/'+key + '_' + eeg_channel[i_channel]+'_meannorm')
        # plot_psd_ratio(subject_band_array, subject_band_array_nor, key)  # plot everyone eeg band ratio
