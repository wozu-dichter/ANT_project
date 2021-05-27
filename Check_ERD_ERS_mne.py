import mne
import numpy as np
import _pickle as cPickle
import pandas as pd
import matplotlib.pyplot as plt

x_train1 = np.load("dataset/c95hyt/1/rest_20210121_15_47_08.npy", allow_pickle=True).item()["eeg"]
x_train1 = x_train1.reshape(150000, 16)
rawdata22 = x_train1[:2500, :].T
df = pd.read_csv('channel(10-20).txt')
ch_names = df.name.to_list()
# montage = mne.channels.Montage(pos=pos, ch_names=ch_names, kind='motor-cap', selection=range(64))

info = mne.create_info(
    ch_names=ch_names[:16],
    ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
              'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
    sfreq=500
)

custom_raw = mne.io.RawArray(rawdata22, info)
print(custom_raw)
picks = mne.pick_channels(custom_raw.info['ch_names'],
                          ["C3", "Cz"])
custom_raw.plot_psd(fmax=30)
# plt.show()
scalings = {'eeg': 64}  # Zoom out 64 times
# custom_raw.plot(n_channels=16,
#                 scalings=scalings,
#                 title='Data from arrays',
#                 highpass=1,
#                 lowpass=50,
#                 show=True, block=True)

# custom_raw.plot_psd_topo()
plt.show()
