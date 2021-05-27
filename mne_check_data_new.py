import mne
import numpy as np


def load_npy(path):
    raw_data = np.load(path, allow_pickle=True).item()["eeg"]
    return raw_data


with open("channel(10-20).txt", "r") as f:
    lines = f.readlines()[1:]

channels = []
for line in lines:
    line = line.split(",")[0]
    channels.append(line)
print("all channels", channels)

num_channels = len(channels)
ch_types = ["eeg"] * num_channels


raw_data = load_npy("./dataset/c95ccy/1/session_20210127_09_41_52.npy")
raw_data = np.reshape(raw_data, (-1, num_channels)).T

info = mne.create_info(ch_names=channels, sfreq=500, ch_types=ch_types)
custom_raw = mne.io.RawArray(raw_data, info)
# custom_raw.plot(n_channels=8,
#                 scalings={"eeg": 100},
#                 title="Data from arrays",
#                 show=True, block=True)
custom_raw.plot_psd()
