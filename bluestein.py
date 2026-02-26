import numpy as np
import matplotlib.pyplot as plt
import time

def bluestein_fft(x):
    N = len(x)

    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * n**2 / N)

    a = x*chirp

    m = np.arange(2*N-1)
    b = np.exp(1j*np.pi*(m-(N-1))**2/N)

    M = 1 << (2*N-1).bit_length()
    A = np.fft.fft(a,M)
    B = np.fft.fft(b,M)
    c = np.fft.ifft(A*B)[:N]

    return c*chirp


x = np.random.rand(10000) + 1j*np.random.rand(10000)

start1 = time.time()
# Bluestein FFT
Xb = bluestein_fft(x)[2:-2]
end1 = time.time()

start2 = time.time()
# Compare with numpy’s FFT
Xn = np.fft.fft(x)[2:-2]
end2 = time.time()

print(end1 - start1)
print(end2 - start2)
print("Error:", np.max(np.abs(Xb - Xn)))

plt.plot(np.abs(Xb), label='Bluestein')
plt.plot(np.abs(Xn), label='FFT')
plt.legend()
plt.show()