import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from scipy.fft import fft


def apply_fft(data, sampling_rate):
    available_range = min(len(data), sampling_rate)
    yf = (abs(fft(data)) / available_range)[:available_range // 2]
    xf = np.arange(len(data))[:available_range // 2]
    return xf, yf


def my_stft(data, fs, nperseg, noverlap):
    stride = nperseg - noverlap
    # data = np.hanning(512) * data
    yfs = []
    t = []
    for idx in range(0, len(data), stride):
        d = data[idx:idx + nperseg]
        if len(d) != nperseg:
            break
        t.append(idx / fs)
        xf, yf = apply_fft(d, fs)
        yfs.append(yf)
    yfs = np.array(yfs).T
    t = np.array(t)
    return xf, t, yfs


sampling_rate = 1024
t = np.arange(0, 10.0, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300
data = np.piecewise(t, [t < 10, t < 8, t < 3],
                    [lambda t: np.sin(2 * np.pi * f1 * t),
                     lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])

plt.subplots_adjust(hspace=0.5)
f, t, zxx = stft(data, fs=sampling_rate, nperseg=50)
plt.subplot(211)
plt.pcolormesh(t, f, np.abs(zxx), shading=None)
plt.title("STFT Magnitude")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")

xf, tt, yfs = my_stft(data, fs=sampling_rate, nperseg=1024, noverlap=900)
plt.subplot(212)
plt.pcolormesh(tt, xf, yfs, shading="auto")
plt.title("My STFT")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()
a = 0

plt.show()

# xf, yf = apply_fft(data, sampling_rate)
# plt.plot(xf, yf, "b")
# plt.show()
