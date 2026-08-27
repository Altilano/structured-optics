import numpy as np
import matplotlib.pyplot as plt
import time


def bluestein_1d(
    f,
    dx,
    dk,
    N_out=None,
    x0=0.0,
    k0=0.0,
):
    """
    Generalized 1D Fourier transform using Bluestein's algorithm.

    Computes

        F(k_m) = ∫ f(x) exp(-i k_m x) dx

    approximately as

        F(k_m) = dx * sum_n f(x_n) exp(-i k_m x_n)

    where

        x_n = x0 + n*dx
        k_m = k0 + m*dk

    Parameters
    ----------
    f : ndarray
        Input array. Transform is performed along the last axis.

    dx : float
        Input spatial sampling.

    dk : float
        Output k-space sampling in rad / unit-x.

    N_out : int, optional
        Number of output samples.

    x0 : float
        Coordinate of first input sample.

    k0 : float
        First output k coordinate.

    Returns
    -------
    F : ndarray
        Fourier transform.
    """

    f = np.asarray(f)

    N = f.shape[-1]

    if N_out is None:
        N_out = N

    if N < 1:
        raise ValueError("Input cannot be empty.")

    if N_out < 1:
        raise ValueError("N_out must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if dk == 0:
        raise ValueError("dk cannot be zero.")

    # ------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------

    n = np.arange(N)
    m = np.arange(N_out)

    # ------------------------------------------------------------
    # Basic phase coefficient
    # ------------------------------------------------------------

    a = dx * dk

    # ------------------------------------------------------------
    # Chirps
    #
    # exp(-i a n m)
    #
    # =
    #
    # exp(-i a n²/2)
    # exp(-i a m²/2)
    # exp(+i a(n-m)²/2)
    # ------------------------------------------------------------

    chirp_n = np.exp(
        -0.5j * a * n**2
    )

    chirp_m = np.exp(
        -0.5j * a * m**2
    )

    # ------------------------------------------------------------
    # Origin phases
    # ------------------------------------------------------------

    input_phase = np.exp(
        -1j * k0 * dx * n
    )

    output_phase = np.exp(
        -1j * x0 * dk * m
    )

    global_phase = np.exp(
        -1j * k0 * x0
    )

    # ------------------------------------------------------------
    # Convolution input
    # ------------------------------------------------------------

    a_input = (
        f
        * input_phase
        * chirp_n
    )

    # ------------------------------------------------------------
    # Convolution kernel
    #
    # q = n - m
    # ------------------------------------------------------------

    q = np.arange(
        -(N_out - 1),
        N
    )

    kernel = np.exp(
        +0.5j * a * q**2
    )

    # ------------------------------------------------------------
    # Linear convolution
    # ------------------------------------------------------------

    L = N + N_out - 1

    # Next power of two
    P = 1 << (L - 1).bit_length()

    A = np.fft.fft(
        a_input,
        P,
        axis=-1
    )

    B = np.fft.fft(
        kernel,
        P
    )

    B = B.reshape(
        (1,) * (f.ndim - 1) + (P,)
    )

    convolution = np.fft.ifft(
        A * B,
        axis=-1
    )

    # ------------------------------------------------------------
    # Extract desired samples
    # ------------------------------------------------------------

    convolution = np.take(
        convolution,
        np.arange(
            N - 1,
            N - 1 + N_out
        ),
        axis=-1
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    F = (global_phase* output_phase* chirp_m* convolution)
    return F





x = np.random.rand(100) + 1j*np.random.rand(100)

N = 10000

t = np.arange(N) * (2*np.pi/N)
x = np.cos(8*2*np.pi*t)
dx = t[1]- t[0]
dk = 2*np.pi/(dx)/N
start1 = time.time()
# Bluestein FFT
#Xb = bluestein_1d(x,dx, dk, N_out=3*N)
dk = 0.0001

kx0 = -50
N_out = 60000

Xb = bluestein_1d(
    x,
    dx=dx,
    dk=dk,
    N_out=N_out,
    k0=kx0
)
end1 = time.time()

start2 = time.time()
# Compare with numpy’s FFT
Xn = np.fft.fft(x)
end2 = time.time()

k = np.arange(N)*dk
print(end1 - start1)
print(end2 - start2)
#print("Error:", np.max(np.abs(Xb - Xn)))

plt.plot(np.abs(Xb), 'o', label='Bluestein')
#plt.plot(k, np.abs(Xn), '+', label='FFT')
plt.legend()
plt.show()