import numpy as np


def fft_bluestein(f, dx, dk, D_out=None, x0=0.0, k0=0.0, inverse=False):
    """
    Generalized 1D Fourier transform using Bluestein's algorithm.

    Forward:
        F(k) = ∫ f(x) exp(-i k x) dx

    Inverse:
        f(x) = 1/(2π) ∫ F(k) exp(+i k x) dk
    """

    f = np.asarray(f)
    D = f.shape[-1]

    if D_out is None:
        D_out = D
    if D < 1:
        raise ValueError("Input cannot be empty.")
    if D_out < 1:
        raise ValueError("D_out must be positive.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if dk == 0:
        raise ValueError("dk cannot be zero.")

    n = np.arange(D)
    m = np.arange(D_out)
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
        q = np.arange(-(D - 1),D_out)
        kernel = np.exp(+0.5j * a * q**2)

        # Linear convolution length
        L = D + D_out - 1

        # FFT padding
        P = 1 << (L - 1).bit_length()

        A = np.fft.fft(a_input, P, axis=-1)
        B = np.fft.fft(kernel, P)
        B = B.reshape((1,) * (f.ndim - 1) + (P,))
        convolution = np.fft.ifft(A * B, axis=-1)

        # Extract m = 0,...,D_out-1
        convolution = np.take(convolution, np.arange(D - 1, D - 1 + D_out), axis=-1)

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
        q = np.arange(-(D - 1), D_out)
        kernel = np.exp(-0.5j * a * q**2)
        L = D + D_out - 1
        P = 1 << (L - 1).bit_length()
        A = np.fft.fft(a_input, P, axis=-1)
        B = np.fft.fft(kernel, P)
        B = B.reshape((1,) * (f.ndim - 1) + (P,))
        convolution = np.fft.ifft(A * B, axis=-1)
        convolution = np.take(convolution, np.arange(D - 1, D - 1 + D_out), axis=-1)
        result = (dk/(2*np.pi) * global_phase * output_phase * chirp_m * convolution)
        x = x0 + m * dx
        return result, x

def fft2_bluestein(field, dx, dy, dkx, dky, Dx_out=None, Dy_out=None, x0=0.0, y0=0.0, kx0=0.0, ky0=0.0):
    """
        Generalized 2D Fourier transform using Bluestein's algorithm. Performs fft_bluestein twice.
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("bluestein_fft2 expects a 2D array.")

    Dy, Dx = field.shape
    if Dx_out is None:
        Dx_out = Dx
    if Dy_out is None:
        Dy_out = Dy

    # x transform
    F, kx = fft_bluestein(field, dx=dx, dk=dkx, D_out=Dx_out, x0=x0, k0=kx0)

    # y transform
    F = np.swapaxes(F, -1, -2)
    F, ky = fft_bluestein(F, dx=dy, dk=dky, D_out=Dy_out, x0=y0, k0=ky0)
    F = np.swapaxes(F, -1, -2)
    return F, kx, ky

