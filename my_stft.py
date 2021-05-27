# import numpy as np
# from scipy.fft import fft
#
# sampling_rate = 1024
# t = np.arange(0, 1.0, 1.0 / sampling_rate)
# f1, f2, f3 = 100, 200, 300
# data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
#                     [lambda t: np.sin(2 * np.pi * f1 * t),
#                      lambda t: np.sin(2 * np.pi * f2 * t),
#                      lambda t: np.sin(2 * np.pi * f3 * t)])
#
# print("shape of data : {}".format(data.shape))
# aa = fft(data)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq
#
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N * T, N, endpoint=False)
# y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N // 2]
# plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
# plt.grid()
# plt.show()

import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# 取樣點選擇1400個，因為設定的訊號頻率分量最高為600赫茲，根據取樣定理知取樣頻率要大於訊號頻率2倍，所以這裡設定取樣頻率為1400赫茲（即一秒內有1400個取樣點，一樣意思的）
x = np.linspace(0, 1, 1400)
# 設定需要取樣的訊號，頻率分量有180，390和600
y = 7 * np.sin(2 * np.pi * 180 * x) + 2.8 * np.sin(2 * np.pi * 390 * x) + 5.1 * np.sin(2 * np.pi * 600 * x)

yy = fft(y)  # 快速傅立葉變換

yf = abs(fft(y))  # 取絕對值
yf1 = abs(fft(y)) / len(x)  # 歸一化處理
yf2 = yf1[range(int(len(x) / 2))]  # 由於對稱性，只取一半區間

xf = np.arange(len(y))  # 頻率
xf1 = xf
xf2 = xf[range(int(len(x) / 2))]  # 取一半區間

plt.subplot(221)
plt.plot(x[0:50], y[0:50])
plt.title("Original wave")

plt.subplot(222)
plt.plot(xf, yf, "r")
plt.title("FFT of Mixed wave(two sides frequency range)", fontsize=7, color="#7A378B")  # 注意這裡的顏色可以查詢顏色程式碼表

plt.subplot(223)
plt.plot(xf1, yf1, "g")
plt.title("FFT of Mixed wave(normalization)", fontsize=9, color="r")

plt.subplot(224)
plt.plot(xf2, yf2, "b")
plt.title("FFT of Mixed wave)", fontsize=10, color="#F08080")

plt.show()
