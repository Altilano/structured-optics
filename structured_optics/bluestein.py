import numpy as np


def fft_bluestein(f, dx, dk, D_out=None, x0=0.0, k0=0.0, inverse=False):
    """
    Generalized 1D Fourier transform using Bluestein's algorithm
    (chirp-z transform).
 
    Unlike a standard FFT, this evaluates the continuous Fourier
    integral on arbitrary, independently chosen input/output sample
    spacings (`dx`, `dk`) and offsets (`x0`, `k0`), and can change the
    number of samples from `D` (input) to `D_out` (output). It works by
    rewriting the transform as a linear convolution (via the standard
    Bluestein/chirp-z trick) and evaluating that convolution with zero-
    padded FFTs, so it costs O((D + D_out) log(D + D_out)) rather than
    the O(D * D_out) of a direct sum.
 
    Forward:
        F(k) = integral of f(x) * exp(-i k x) dx
 
    Inverse:
        f(x) = 1/(2*pi) * integral of F(k) * exp(+i k x) dk
 
    Parameters
    ----------
    f : array_like
        Input samples. The transform is taken along the last axis, so
        `f` may have leading batch dimensions (shape `(..., D)`).
    dx : float
        Input sample spacing. Must be positive. In inverse mode, this
        is the desired *output* spacing (of `x`).
    dk : float
        Output sample spacing. Must be nonzero. In inverse mode, this
        is the *input* spacing (of `k`/`F`).
    D_out : int, optional
        Number of output samples. Defaults to `D` (the input length,
        i.e. `f.shape[-1]`), giving a same-length transform.
    x0 : float, optional
        Origin of the `x` grid (forward: input origin; inverse: output
        origin). Defaults to 0.0.
    k0 : float, optional
        Origin of the `k` grid (forward: output origin; inverse: input
        origin). Defaults to 0.0.
    inverse : bool, optional
        If False (default), compute the forward transform
        `f(x) -> F(k)`. If True, compute the inverse transform
        `F(k) -> f(x)` (with `f` here holding the input spectrum
        samples).
 
    Returns
    -------
    result : np.ndarray
        Forward mode: the transformed array `F`, of shape
        `(..., D_out)`.
        Inverse mode: the reconstructed array `f`, of shape
        `(..., D_out)`.
    coords : np.ndarray
        Forward mode: the output `k` grid, `k0 + m*dk` for
        `m = 0, ..., D_out-1`.
        Inverse mode: the output `x` grid, `x0 + m*dx` for
        `m = 0, ..., D_out-1`.
 
    Raises
    ------
    ValueError
        If the input is empty, `D_out` is not positive, `dx` is not
        positive, or `dk` is zero.
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
    Generalized 2D (forward) Fourier transform using Bluestein's
    algorithm, applied separably along each axis.
 
    Performs `fft_bluestein` along the last axis (x), then along the
    remaining axis (y), i.e. a separable 2D chirp-z transform with
    independently chosen sample spacing, sample count, and origin for
    each axis.
 
    Parameters
    ----------
    field : array_like
        2D input array of shape `(Dy, Dx)`.
    dx : float
        Input sample spacing along x.
    dy : float
        Input sample spacing along y.
    dkx : float
        Output sample spacing along kx. Must be nonzero.
    dky : float
        Output sample spacing along ky. Must be nonzero.
    Dx_out : int, optional
        Number of output samples along x. Defaults to `Dx`
        (`field.shape[1]`).
    Dy_out : int, optional
        Number of output samples along y. Defaults to `Dy`
        (`field.shape[0]`).
    x0 : float, optional
        Origin of the input x grid. Defaults to 0.0.
    y0 : float, optional
        Origin of the input y grid. Defaults to 0.0.
    kx0 : float, optional
        Origin of the output kx grid. Defaults to 0.0.
    ky0 : float, optional
        Origin of the output ky grid. Defaults to 0.0.
 
    Returns
    -------
    F : np.ndarray
        Transformed array, of shape `(Dy_out, Dx_out)`.
    kx : np.ndarray
        Output kx grid, `kx0 + m*dkx` for `m = 0, ..., Dx_out-1`.
    ky : np.ndarray
        Output ky grid, `ky0 + m*dky` for `m = 0, ..., Dy_out-1`.
 
    Raises
    ------
    ValueError
        If `field` is not 2-dimensional, or if any of the underlying
        `fft_bluestein` calls raise (e.g. non-positive spacing, zero
        `dk`).
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

