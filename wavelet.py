import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.font_manager import FontProperties

chinese_font = FontProperties()
sampling_rate = 500
t = np.arange(0, 5, 5*sampling_rate)
f1 = 100
f2 = 200
f3 = 300
data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: np.sin(2 * np.pi * f1 * t),
                     lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])

wavename = "gaus8"
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel("time(seconds)", fontproperties=chinese_font)
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel("frequency(Hz)", fontproperties=chinese_font)
plt.xlabel("time(seconds)", fontproperties=chinese_font)
plt.subplots_adjust(hspace=0.4)
plt.show()
