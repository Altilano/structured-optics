import numpy as np


def fft_bluestein(f, dx, dk, N_out=None, x0=0.0, k0=0.0, inverse=False):
    """
    Generalized 1D Fourier transform using Bluestein's algorithm.

    Forward:
        F(k) = ∫ f(x) exp(-i k x) dx

    Inverse:
        f(x) = 1/(2π) ∫ F(k) exp(+i k x) dk
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

    n = np.arange(N)
    m = np.arange(N_out)

    a = dx * dk

    # FORWARD
    if not inverse:

        # Chirps
        chirp_n = np.exp(-0.5j * a * n**2)

        chirp_m = np.exp(-0.5j * a * m**2)

        # Coordinate-origin phases
        input_phase = np.exp(-1j * k0 * dx * n)

        output_phase = np.exp(-1j * x0 * dk * m)

        global_phase = np.exp(-1j * k0 * x0)

        # Input chirp
        a_input = (f * input_phase * chirp_n)

        # Convolution kernel
        q = np.arange(-(N - 1),N_out)

        kernel = np.exp(+0.5j * a * q**2)

        # Linear convolution length
        L = N + N_out - 1

        # FFT padding
        P = 1 << (L - 1).bit_length()

        A = np.fft.fft(a_input, P, axis=-1)

        B = np.fft.fft(kernel, P)

        B = B.reshape((1,) * (f.ndim - 1) + (P,))

        convolution = np.fft.ifft(A * B, axis=-1)

        # Extract m = 0,...,N_out-1
        convolution = np.take(convolution, np.arange(N - 1, N - 1 + N_out), axis=-1)

        # Final result
        F = (dx * global_phase * output_phase * chirp_m * convolution)

        k = k0 + m * dk

        return F, k


    # INVERSE
    else:

        # Positive exponential
        chirp_n = np.exp(+0.5j * a * n**2)

        chirp_m = np.exp(+0.5j * a * m**2)

        input_phase = np.exp(+1j * x0 * dk * n)

        output_phase = np.exp(+1j * k0 * dx * m)

        global_phase = np.exp(+1j * k0 * x0)

        a_input = (f * input_phase * chirp_n)

        # Correct convolution kernel
        q = np.arange(-(N - 1), N_out)

        kernel = np.exp(-0.5j * a * q**2)

        L = N + N_out - 1

        P = 1 << (L - 1).bit_length()

        A = np.fft.fft(a_input, P, axis=-1)

        B = np.fft.fft(kernel, P)

        B = B.reshape((1,) * (f.ndim - 1) + (P,))

        convolution = np.fft.ifft(A * B, axis=-1)

        convolution = np.take(convolution, np.arange(N - 1, N - 1 + N_out), axis=-1)

        result = (dk/(2*np.pi) * global_phase * output_phase * chirp_m * convolution)

        x = x0 + m * dx

        return result, x

def fft2_bluestein(field, dx, dy, dkx, dky, Nx_out=None, Ny_out=None, x0=0.0, y0=0.0, kx0=0.0, ky0=0.0):
    field = np.asarray(field)

    if field.ndim != 2:
        raise ValueError("bluestein_fft2 expects a 2D array.")

    Ny, Nx = field.shape

    if Nx_out is None:
        Nx_out = Nx

    if Ny_out is None:
        Ny_out = Ny

    # x transform
    F, kx = fft_bluestein(field, dx=dx, dk=dkx, N_out=Nx_out, x0=x0, k0=kx0)

    # y transform
    F = np.swapaxes(F, -1, -2)

    F, ky = fft_bluestein(F, dx=dy, dk=dky, N_out=Ny_out, x0=y0, k0=ky0)

    F = np.swapaxes(F, -1, -2)

    return F, kx, ky

