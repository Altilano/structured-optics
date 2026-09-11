import numpy as np
from scipy import fft
from structured_optics.bluestein import *


def propagate_fresnel_conv(Beam, z):                                     
    #Propagate the field by a distance z using fresnel integral. convolution method
    k = Beam.k()
    prop = np.exp(-1j * z * (Beam.kx ** 2 + Beam.ky ** 2) / (2 * k) + 1j*k*z)
    Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
    Beam.field = fft.ifft2(prop*Beam.fourier_field, axes=(-2,-1))
    return Beam

def propagate_fresnel_fft(Beam, z):
    """
    Propagate a complex scalar field with the Fresnel approximation using
    a single Fourier transform.
 
    Parameters
    ----------
    Beam : object
        Beam class containing all physical information of your beam.
    z : float
        Propagation distance (must be > 0; the Fresnel single-FFT form
        is a forward-propagation formula).
 
    Returns
    -------
    Beam : object
    """
    if z == 0:
        return Beam
 

    Dy, Dx = Beam.field.shape[-2:]
 
    dx = 2*Beam.nix/Beam.Dx
    dy = 2*Beam.niy/Beam.Dy
 
    k = Beam.k()
 
 
    # --- Output grid, fixed by the physics (fx = x_out / (lambda*z)) ---
    dx_out = Beam.lamb * z / (Dx * dx)
    dy_out = Beam.lamb * z / (Dy * dy)
    x_out = (np.arange(Dx) - Dx // 2) * dx_out
    y_out = (np.arange(Dy) - Dy // 2) * dy_out
    X_out, Y_out = np.meshgrid(x_out, y_out)
 
    # --- Input quadratic phase ---
    Q1 = np.exp(1j * k / (2 * z) * (Beam.x**2 + Beam.y**2))
 
    # --- Single FFT (fftshift/ifftshift keep x=0 <-> zero frequency) ---
    U1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Beam.field * Q1, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)) * dx * dy
 
    # --- Output prefactor and quadratic phase ---
    Q2 = (np.exp(1j * k * z) / (1j * Beam.lamb * z)) * \
         np.exp(1j * k / (2 * z) * (X_out**2 + Y_out**2))
 
    field_z = Q2 * U1

    if dx_out < 0:
        x_out = x_out[::-1]
        field_z = field_z[..., :, ::-1]
    if dy_out < 0:
        y_out = y_out[::-1]
        field_z = field_z[..., ::-1, :]

    Beam.field = field_z
    Beam.x, Beam.y = np.meshgrid(x_out, y_out, sparse=True)
 
    return Beam

def propagate_bluestein(Beam, z, x_out_range, y_out_range, Dx_out, Dy_out):
    """
    Single-FFT Fresnel propagation with an arbitrary, independently
    chosen output grid, using the Bluestein (chirp-z) transform.
 
    Parameters
    ----------
    Beam : object
        Beam class containing all physical information of your beam.
    z : float
        Propagation distance. Must be nonzero; negative z back-propagates.
    x_out_range, y_out_range : (float, float)
        (min, max) of the desired OUTPUT region, in the same units as x, y.
    Dx_out, Dy_out : int
        Number of output samples along x and y. Independent of the input
        sample count, spacing, or of lambda*z/(N*dx) -- choose whatever
        resolution/region you need.
 
    Returns
    -------
    Beam : object
    """
    if z == 0:
        raise ValueError("Bluestein propagation requires z != 0.")
    if Dx_out < 2 or Dy_out < 2:
        raise ValueError("Dx_out and Dy_out must each be >= 2.")
 
    x = np.asarray(Beam.x).ravel()
    y = np.asarray(Beam.y).ravel()
 
    dx_in = 2*Beam.nix/Beam.Dx
    dy_in = 2*Beam.niy/Beam.Dy
    x0_in = x[0]
    y0_in = y[0]
 
    x_out = np.linspace(x_out_range[0], x_out_range[1], Dx_out)
    y_out = np.linspace(y_out_range[0], y_out_range[1], Dy_out)
    dx_out = x_out[1] - x_out[0]
    dy_out = y_out[1] - y_out[0]
    x0_out = x_out[0]
    y0_out = y_out[0]
 

    k = Beam.k()
 
    X, Y = np.meshgrid(x, y)                   # input grid, (Dy, Dx)
    X_out, Y_out = np.meshgrid(x_out, y_out)   # output grid, (Dy_out, Dx_out)
 
    # --- Input quadratic phase ---
    Q1 = np.exp(1j * k / (2 * z) * (X**2 + Y**2))
    g = Beam.field * Q1   # shape (..., Dy, Dx)
 
    # --- Map desired output coordinates to CZT frequency-grid parameters ---
    # cross term: exp[-i (k/z) x_out * x0]  ->  K = (k/z) * x_out
    dKx = (k / z) * dx_out
    Kx0 = (k / z) * x0_out
    dKy = (k / z) * dy_out
    Ky0 = (k / z) * y0_out
 
    # --- CZT along x (last axis) ---
    Gx, _ = fft_bluestein(g, dx=dx_in, dk=dKx, D_out=Dx_out,
                           x0=x0_in, k0=Kx0, inverse=False)
 
    # --- CZT along y (swap so y is the last axis, then swap back) ---
    Gx = np.swapaxes(Gx, -1, -2)
    Gxy, _ = fft_bluestein(Gx, dx=dy_in, dk=dKy, D_out=Dy_out,
                            x0=y0_in, k0=Ky0, inverse=False)
    Gxy = np.swapaxes(Gxy, -1, -2)   # back to (..., Dy_out, Dx_out)
 
    # --- Output prefactor and quadratic phase ---
    Q2 = (np.exp(1j * k * z) / (1j * Beam.lamb * z)) * \
         np.exp(1j * k / (2 * z) * (X_out**2 + Y_out**2))
 
    Beam.field = Q2 * Gxy
    Beam.x, Beam.y = np.meshgrid(x_out, y_out, sparse=True)
    Beam.Dx = Dx_out
    Beam.Dy = Dy_out
    Beam.nix = (x_out_range[1] - x_out_range[0])/2
    Beam.niy = (y_out_range[1] - y_out_range[0])/2
    Beam.kx, Beam.ky = np.meshgrid(2*np.pi*fft.fftfreq(Beam.Dx, 2*Beam.nix/(Beam.Dx-1)), 2*np.pi*fft.fftfreq(Beam.Dy, 2*Beam.niy/(Beam.Dy-1)), sparse=True)
 
    return Beam



def propagate_angular_spectrum(Beam, z, evanescent=False):
    #Propagate the field by a distance z using Angular Spectrum method.
    k = Beam.k()
    kz2 = k**2 - Beam.kx**2 - Beam.ky**2
    kz = np.sqrt(kz2.astype(complex))
    if not evanescent:
        not_evanescent = kz2 >= 0
        kz = kz*not_evanescent
    prop = np.exp(1j*kz*z + 1j*k*z)
    Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
    Beam.field = fft.ifft2(prop*Beam.fourier_field, axes=(-2,-1))
    return Beam


def propagate_incoherent(Beam, z):
    #Propagates the INTENSITY by a distance z using incoherent propagation. Loses phase information.
    #without pupil function doesn't seems to work
    k = Beam.k()
    if z != 0:
        h2 = np.ones_like(Beam.field)/(Beam.lamb*z)**2
    else:
        h2 = np.ones_like(Beam.field)
    h2_fourier = fft.fft2(h2)
    I_fourier = fft.fft2(Beam.int_profile())
    Beam.field = np.sqrt(fft.ifft2(h2_fourier*I_fourier))
    return Beam
    
    
def propagate_fraunhofer(Beam, z):
    #Propagate the field by a distance z using fraunhofer integral. Changes XY grid.

    if z < 2/Beam.lamb*Beam.waist**2:
        print("z small. Fraunhofer is probably not a good aproximation.") 
    if z != 0:
        k = Beam.k()
        dx = Beam.x[0,1] - Beam.x[0,0]
        dy = Beam.y[1,0] - Beam.y[0,0]
        fx = np.fft.fftshift(np.fft.fftfreq(Beam.Dx, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(Beam.Dy, d=dy))
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        Beam.x = Beam.lamb * z * FX
        Beam.y = Beam.lamb * z * FY
        Beam.nix = np.max(Beam.x)
        Beam.niy = np.max(Beam.y)
        C = np.exp(1j*k*z)/(1j*Beam.lamb*z)*np.exp(1j*k*(Beam.x**2 + Beam.y**2)/(2*z))
        Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
        Beam.field = C*fft.fftshift(Beam.fourier_field)*dx*dy
    return Beam







def suggest_propagation_method(Beam, z, aperture_sigma_factor=3.0,
                                fraunhofer_threshold=0.1, verbose=True):
    """
    Recommend a propagation method for Beam.propagate(z, method=...).
 
    Parameters
    ----------
    Beam : object
        Beam instance.
    z : float
        Intended propagation distance.
    aperture_sigma_factor : float, optional
        With sigma = Beam.std()[0] the effective aperture half-width
        used for the Fraunhofer/paraxial checks is
        aperture_sigma_factor * sigma (default 3.0, i.e. ~99.7% of the
        energy for a Gaussian profile). Only affects the far-field/paraxial
        checks -- the transfer-function sampling check (z_crit) always
        uses the full grid, since that's a property of the numerical grid,
        not of the beam itself.
    fraunhofer_threshold : float, optional
        Fresnel number below which Fraunhofer is considered valid
        (default 0.1, i.e. within a few percent of the true far field).
    verbose : bool, optional
        If True, print the diagnostic numbers behind the recommendation.
 
    Returns
    -------
    method : str
        One of: 'none', 'AS', 'fres_c', 'fraun'.
    info : dict
        The underlying numbers (z_crit, fresnel_number, z_paraxial_min,
        aperture size used, and where that size came from) in case you
        want to make the decision yourself.
    """
    if z == 0:
        if verbose:
            print("z = 0: no propagation needed.")
        return 'none', {}
 
    abs_z = abs(z)
    lam = Beam.lamb
 
    dx = 2 * Beam.nix / Beam.Dx
    dy = 2 * Beam.niy / Beam.Dy
    Dx, Dy = Beam.Dx, Beam.Dy
 
    # (1) transfer-function critical distance (smaller axis is the binding
    #     one) -- always based on the grid, not the beam
    z_crit_x = Dx * dx**2 / lam
    z_crit_y = Dy * dy**2 / lam
    z_crit = min(z_crit_x, z_crit_y)
 
    # (2) effective aperture size for the far-field/paraxial checks:
    #     prefer the beam's own extent (via Beam.std()) over the full
    #     simulation window, since a beam much smaller than the window
    #     reaches the far field far sooner than the window would suggest.
    ax = ay = None
    size_source = "simulation window half-width (nix, niy)"
    if hasattr(Beam, 'std'):
        try:
            sigma_x, sigma_y = Beam.std()
            if sigma_x > 0 and sigma_y > 0:
                ax = aperture_sigma_factor * sigma_x
                ay = aperture_sigma_factor * sigma_y
                size_source = f"{aperture_sigma_factor}*sigma from Beam.std()"
        except Exception:
            pass  # fall back to window size below
    if ax is None or ay is None:
        ax, ay = Beam.nix, Beam.niy
 
    # (3) Fraunhofer / Fresnel number
    a2 = ax**2 + ay**2
    fresnel_number = a2 / (lam * abs_z)
 
    # (4) paraxial validity floor (Goodman's criterion), same aperture size
    z_paraxial_min = (np.pi / (4 * lam) * a2**2) ** (1 / 3)
 
    info = {
        'z_crit': z_crit,
        'fresnel_number': fresnel_number,
        'z_paraxial_min': z_paraxial_min,
        'aperture_size': (ax, ay),
        'aperture_size_source': size_source,
    }
 
    if fresnel_number < fraunhofer_threshold:
        method = 'fraun'
    elif abs_z <= z_crit:
        method = 'AS'
    elif abs_z < z_paraxial_min:
        # Too close for paraxial single-step Fresnel to be trustworthy,
        # yet past z_crit for the transfer-function sampling argument --
        # fall back to the exact angular spectrum method.
        method = 'AS'
    else:
        method = 'fres_c'
 
    if verbose:
        print(f"z = {z:.4g} m")
        print(f"  aperture size used ({size_source}): ax={ax:.4g} m, ay={ay:.4g} m")
        print(f"  critical (transfer-function) distance z_crit = {z_crit:.4g} m")
        print(f"  Fresnel number                       Nf      = {fresnel_number:.4g}")
        print(f"  paraxial validity floor               z_min   = {z_paraxial_min:.4g} m")
        print(f"  -> recommended method: {method}")
 
    return method, info




def estimate_bluestein_range(Beam, z, n_sigma=5.0):#, include_cross_term=True):
    """
    Estimate the output window to pass to propagate_bluestein so the
    propagated beam is fully contained without edge clipping.
 
    Parameters
    ----------
    Beam : object
        Your Beam instance (uses .field, .x, .y, .kx, .ky, .k()).
    z : float
        Propagation distance you intend to use.
    n_sigma : float, optional
        How many standard deviations of the predicted beam width to
        include on each side (default 5.0 -- for a Gaussian-like profile
        this captures effectively all the energy; increase for beams
        with heavier tails, e.g. Bessel-like profiles).
    include_cross_term : bool, optional
        If True (default), account for any existing wavefront curvature
        (beam already converging/diverging at the current plane). Set to
        False only if you know the current plane is a true waist/collimated
        plane and want to skip the extra derivative computation.
 
    Returns
    -------
    x_out_range, y_out_range : (float, float)
        Suggested ranges for propagate_bluestein.
    info : dict
        Diagnostic quantities: current/predicted sigma, angular spread,
        predicted centroid, etc.
    """
    field = Beam.field
    X, Y = Beam.x, Beam.y   # sparse grids, shapes (1,Dx) and (Dy,1)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    k = Beam.k()
 
    # --- Spatial intensity moments (sum over any polarization axis) ---
    I = np.sum(np.abs(field) ** 2, axis=0)          # shape (Dy, Dx)
    P = np.sum(I) * dx * dy
 
    xbar0 = np.sum(X * I) * dx * dy / P
    ybar0 = np.sum(Y * I) * dx * dy / P
    varx0 = np.sum((X - xbar0) ** 2 * I) * dx * dy / P
    vary0 = np.sum((Y - ybar0) ** 2 * I) * dx * dy / P
    sigma_x0 = np.sqrt(varx0)
    sigma_y0 = np.sqrt(vary0)
 
    # --- Angular-spectrum moments (beam's own Fourier content) ---
    A = fft.fft2(field, axes=(-2, -1))
    S = np.sum(np.abs(A) ** 2, axis=0)              # shape (Dy, Dx)
    S_tot = np.sum(S)
 
    kx_bar0 = np.sum(Beam.kx * S) / S_tot
    ky_bar0 = np.sum(Beam.ky * S) / S_tot
    var_kx0 = np.sum((Beam.kx - kx_bar0) ** 2 * S) / S_tot
    var_ky0 = np.sum((Beam.ky - ky_bar0) ** 2 * S) / S_tot
 
    theta_x_bar0 = kx_bar0 / k
    theta_y_bar0 = ky_bar0 / k
    sigma_thetax0 = np.sqrt(var_kx0) / k
    sigma_thetay0 = np.sqrt(var_ky0) / k
 
    # --- Cross term: existing wavefront curvature, via local phase gradient ---
    cross_x0 = 0.0
    cross_y0 = 0.0
    #if include_cross_term:
    dUdx = fft.ifft2(1j * Beam.kx * A, axes=(-2, -1))
    dUdy = fft.ifft2(1j * Beam.ky * A, axes=(-2, -1))
    # Im(U* dU/dx) = |U|^2 * dphi/dx  (local intensity-weighted phase slope)
    local_px = np.sum(np.conj(field) * dUdx, axis=0).imag
    local_py = np.sum(np.conj(field) * dUdy, axis=0).imag
    cross_x0 = np.sum((X - xbar0) * local_px) * dx * dy / (k * P)
    cross_y0 = np.sum((Y - ybar0) * local_py) * dx * dy / (k * P)
    #end if

    
    # --- Propagate moments to the target z (exact paraxial identity) ---
    xbar_z = xbar0 + z * theta_x_bar0
    ybar_z = ybar0 + z * theta_y_bar0
 
    varx_z = varx0 + 2 * z * cross_x0 + z**2 * sigma_thetax0**2
    vary_z = vary0 + 2 * z * cross_y0 + z**2 * sigma_thetay0**2
    sigma_x_z = np.sqrt(max(varx_z, 0.0))
    sigma_y_z = np.sqrt(max(vary_z, 0.0))
 
    half_x = n_sigma * sigma_x_z
    half_y = n_sigma * sigma_y_z
 
    x_out_range = (xbar_z - half_x, xbar_z + half_x)
    y_out_range = (ybar_z - half_y, ybar_z + half_y)
 
    info = {
        'sigma_x0': sigma_x0, 'sigma_y0': sigma_y0,
        'sigma_thetax0': sigma_thetax0, 'sigma_thetay0': sigma_thetay0,
        'cross_x0': cross_x0, 'cross_y0': cross_y0,
        'xbar0': xbar0, 'ybar0': ybar0,
        'xbar_z': xbar_z, 'ybar_z': ybar_z,
        'sigma_x_z': sigma_x_z, 'sigma_y_z': sigma_y_z,
    }
 
    return x_out_range, y_out_range, info

