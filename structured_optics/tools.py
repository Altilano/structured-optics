import numpy as np

__all__ = ["overlap", "int_overlap", "hg_proj", "lg_proj", "hg_basis", "lg_basis", "bessel_basis"]

#Tools for 
def overlap(first_beam:object, second_beam:object)->complex:
    """
    Compute the (complex) field overlap integral between two beams.

    Only gives a physically meaningful result if both beams share the
    same grid, i.e. the same `nix`/`niy` (half-window size) and the
    same `Dx`/`Dy` (sample count).

    Parameters
    ----------
    first_beam : object
        First Beam instance.
    second_beam : object
        Second Beam instance.

    Returns
    -------
    complex
        The normalized overlap integral
        ``sum(E1 * conj(E2)) * dx * dy / sqrt(P1 * P2)``, where
        `dx = 2*nix/Dx`, `dy = 2*niy/Dy`, and `P1`, `P2` are each
        beam's total power.
    """
    # calculate overlap between two beams, only properly works if ni and D of beams are equal.
    return np.sum(first_beam.field*np.conjugate(second_beam.field))*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/np.sqrt(first_beam.Power()*second_beam.Power())

def int_overlap(first_beam:object, second_beam:object)->float:
    """
    Compute the (real-valued, incoherent) intensity overlap between two
    beams.

    Unlike `overlap`, this compares intensity profiles rather than
    complex fields, so it discards phase information and is insensitive
    to a relative phase between the two beams. Only gives a physically
    meaningful result if both beams share the same grid, i.e. the same
    `nix`/`niy` and `Dx`/`Dy`.

    Parameters
    ----------
    first_beam : object
        First Beam instance.
    second_beam : object
        Second Beam instance.

    Returns
    -------
    float
        The normalized intensity overlap
        ``sum(I1 * I2) * dx * dy / (P1 * P2)``, where `dx = 2*nix/Dx`,
        `dy = 2*niy/Dy`, and `P1`, `P2` are each beam's total power.
    """
    return np.sum(first_beam.int_profile()*second_beam.int_profile())*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/(first_beam.Power()*second_beam.Power())



#Tools for mode decomposition.

def hg_proj(Beam,N):
    """
    Project `Beam` onto the Hermite-Gaussian (HG) basis, up to order N
    in each index.

    Parameters
    ----------
    Beam : object
        Beam instance to decompose. Not modified.
    N : int
        Maximum HG index (inclusive) along both `n` and `m`; the basis
        used has (N+1) x (N+1) modes.

    Returns
    -------
    n, m : np.ndarray
        Meshgrids (each of shape (N+1, N+1)) of the HG mode indices.
    overlaps : np.ndarray
        Complex array of shape (N+1, N+1) where `overlaps[n, m]` is
        `overlap(Beam, HG_{n,m})`, i.e. the complex overlap of `Beam`
        with the HG mode of indices (n, m).
    """
    overlaps = np.zeros((N+1, N+1), dtype='complex')
    auxiliary_beam  = Beam.copy_clean()
    for n in range(N+1):
        for m in range(N+1):
            auxiliary_beam.hg(n,m)
            overlaps[n,m] = overlap(Beam, auxiliary_beam)
    n, m =  np.meshgrid(np.arange(0, N+1, 1), np.arange(0, N+1, 1))
    return n, m, overlaps
    

def lg_proj(Beam, N):
    """
    Project `Beam` onto the Laguerre-Gaussian (LG) basis, with azimuthal
    index l ranging over -N..N and radial index p ranging over
    0..floor(N/2).

    Parameters
    ----------
    Beam : object
        Beam instance to decompose. Not modified.
    N : int
        Controls the range of azimuthal indices (`l` from -N to N) and
        radial indices (`p` from 0 to N//2).

    Returns
    -------
    l, p : np.ndarray
        Meshgrids of the LG mode indices actually used.
    overlaps : np.ndarray
        Complex array of shape (2N+1, N//2 + 1) where
        `overlaps[l+N, p]` is `overlap(Beam, LG_{l,p})`, i.e. the
        complex overlap of `Beam` with the LG mode of indices (l, p).
    """
    overlaps = np.zeros((int(2*N)+1, int(N/2)+1), dtype='complex')
    auxiliary_beam  = Beam.copy_clean()
    for l in np.arange(-N, N+1, 1):
        for p in range(int(N/2)+1):
            auxiliary_beam.lg(l,p)
            overlaps[l+N,p] = overlap(Beam, auxiliary_beam)
    p, l = np.meshgrid(np.arange(0, int(N/2)+1, 1), np.arange(-N, N+1, 1))
    return l, p, overlaps
    
def hg_basis(Beam, N, waist = None, norm1 = False):
    """
    Build an array of Hermite-Gaussian basis mode fields, for all
    (n, m) with n, m in 0..N.

    Parameters
    ----------
    Beam : object
        Template Beam instance (grid, wavelength, etc. are taken from
        it via `Beam.copy_clean()`); not modified.
    N : int
        Maximum HG index (inclusive) along both `n` and `m`.
    waist : float, optional
        If given, overrides the auxiliary beam's waist before
        generating each mode. If `None` (default), the template
        beam's own waist is used.
    norm1 : bool, optional
        If True, the whole basis array is divided by its maximum value
        (a single global normalization, not a per-mode one). Defaults
        to False.

    Returns
    -------
    np.ndarray
        Complex array of shape (N+1, N+1, Beam.Dy, Beam.Dx), where
        `basis[n, m]` is the transverse `Ex` field of the HG(n, m)
        mode.
    """
    aux = Beam.copy_clean()
    if waist != None:
        aux.waist = waist
    basis = np.empty((N+1, N+1, Beam.Dy, Beam.Dx), dtype = 'complex')
    for n in range(N+1):
        for m in range(N+1):
            aux.hg(n, m)
            basis[n, m] = aux.Ex
    if norm1 == True:
        basis = basis/np.max(basis)
    return basis

def lg_basis(Beam, N, waist=None, norm1=False):
    """
    Build an array of Laguerre-Gaussian basis mode fields, indexed on
    an (n, m) grid (n, m in 0..N) rather than directly by (l, p).

    For each (n, m), the LG mode used is `LG(l, p)` with
    `l = m - n` and `p = min(n, m)` -- i.e. moving along a row/column
    of the (n, m) grid sweeps through LG modes of varying azimuthal
    index at a fixed "diagonal" relationship to the radial index. This
    indexing is a convenience for building a square basis array; see
    the `l`, `p` computed inside the loop if you need the actual LG
    indices for a given (n, m).

    Parameters
    ----------
    Beam : object
        Template Beam instance (grid, wavelength, etc. are taken from
        it via `Beam.copy_clean()`); not modified.
    N : int
        Maximum index (inclusive) along both `n` and `m`.
    waist : float, optional
        If given, overrides the auxiliary beam's waist before
        generating each mode. If `None` (default), the template
        beam's own waist is used.
    norm1 : bool, optional
        If True, the whole basis array is divided by its maximum value
        (a single global normalization, not a per-mode one). Defaults
        to False.

    Returns
    -------
    np.ndarray
        Complex array of shape (N+1, N+1, Beam.Dy, Beam.Dx), where
        `basis[n, m]` is the transverse `Ex` field of the
        `LG(l=m-n, p=min(n,m))` mode.
    """
    aux = Beam.copy_clean()
    if waist != None:
        aux.waist = waist
    basis = np.empty((N+1, N+1, Beam.Dy, Beam.Dx), dtype = 'complex')
    for n in range(N+1):
        for m in range(N+1):
            l = m-n
            p = min(n,m)
            aux.lg(l, p)
            basis[n, m] = aux.Ex
    if norm1 == True:
        basis = basis/np.max(basis)
    return basis

def bessel_basis(Beam, Nmax, waist=None, norm1=False):
    """
    Build an array of Bessel-beam basis fields, for orders
    n = -Nmax..Nmax.

    Parameters
    ----------
    Beam : object
        Template Beam instance (grid, wavelength, etc. are taken from
        it via `Beam.copy_clean()`); not modified.
    Nmax : int
        Maximum Bessel order magnitude; orders from `-Nmax` to `Nmax`
        (inclusive) are generated, i.e. `2*Nmax + 1` modes total.
    waist : float, optional
        If given, overrides the auxiliary beam's waist before
        generating each mode. If `None` (default), the template
        beam's own waist is used.
    norm1 : bool, optional
        If True, the whole basis array is divided by the maximum of
        its absolute value (a single global normalization, not a
        per-mode one). Defaults to False.

    Returns
    -------
    np.ndarray
        Complex array of shape (2*Nmax+1, Beam.Dy, Beam.Dx), where
        `basis[n]` is the full field of the Bessel beam of order
        `n - Nmax`.
    """
    aux = Beam.copy_clean()
    if waist is not None:
        aux.waist = waist
    # basis[n] = Bessel beam of order n
    basis = np.empty((2*Nmax+1, aux.Dy, aux.Dx), dtype='complex')
    for n in range(2*Nmax+1):
        aux.bessel(n-Nmax)   # or aux.bessel(n, theta=...)
        basis[n] = aux.field
    if norm1:
        basis = basis / np.max(np.abs(basis))
    return basis

def _build_from_coefs_and_basis(Beam, coefs, basis, pol_index = 0):
    """
    Reconstruct a field as a weighted sum of basis modes and store it
    into `Beam.field`.

    Internal helper (not in `__all__`); used to turn a modal
    decomposition (e.g. from `hg_proj`/`lg_proj`) plus its basis
    (e.g. from `hg_basis`/`lg_basis`/`bessel_basis`) back into an
    actual field.

    Parameters
    ----------
    Beam : object
        Beam instance whose `.field` will be overwritten with the
        reconstructed field.
    coefs : np.ndarray
        2D array of complex coefficients, shape (N, N), one per basis
        mode.
    basis : np.ndarray
        Array of basis mode fields, shape (N, N, Dy, Dx), matching
        `coefs` in its first two dimensions (e.g. as returned by
        `hg_basis` or `lg_basis`).
    pol_index : int, optional
        Polarization/field-component index of `Beam.field` (and of the
        internal auxiliary field) that the reconstructed sum is
        written into. Defaults to 0.

    Returns
    -------
    Beam : object
        `Beam`, with `Beam.field` set to the auxiliary field after
        accumulating `sum(coefs[i, j] * basis[i, j])` into component
        `pol_index`.
    """
    N = len(coefs)
    aux = Beam.copy_clean()
    for i in range(N):
        for j in range(N):
            aux.field[pol_index] = aux.field[pol_index] + coefs[i,j] * basis[i,j]
    Beam.field = aux.field
    return Beam