from utils import overlap
import numpy as np



def hg_proj(Beam,N, completeness = False):
    overlaps = np.zeros((N+1, N+1), dtype='complex')
    auxiliary_beam  = Beam.copy_clean()
    for n in range(N+1):
        for m in range(N+1):
            auxiliary_beam.hg(n,m)
            overlaps[n,m] = overlap(Beam, auxiliary_beam)
    n, m =  np.meshgrid(np.arange(0, N+1, 1), np.arange(0, N+1, 1))
    comp = np.sum(np.abs(overlaps)**2)
    if completeness == True:
        return n, m, overlaps, comp
    else:
        return n, m, overlaps
    

def lg_proj(Beam, N, completeness = False):
    overlaps = np.zeros((int(2*N)+1, int(N/2)+1), dtype='complex')
    auxiliary_beam  = Beam.copy_clean()
    for l in np.arange(-N, N+1, 1):
        for p in range(int(N/2)+1):
            auxiliary_beam.lg(l,p)
            overlaps[l+N,p] = overlap(Beam, auxiliary_beam)
    p, l = np.meshgrid(np.arange(0, int(N/2)+1, 1), np.arange(-N, N+1, 1))
    comp = np.sum(np.abs(overlaps)**2)
    if completeness == True:
        return l, p, overlaps, comp
    else:
        return l, p, overlaps
    
def hg_basis(Beam, N, waist = None, norm1 = False):
    aux = Beam.copy_clean()
    if waist != None:
        aux.waist = waist
    basis = np.empty((N+1, N+1, Beam.Dy, Beam.Dx), dtype = 'complex')
    for n in range(N+1):
        for m in range(N+1):
            aux.hg(n, m)
            basis[n, m] = aux.field
    if norm1 == True:
        basis = basis/np.max(basis)
    return basis

def lg_basis(Beam, N, waist=None, norm1=False):
    aux = Beam.copy_clean()
    if waist != None:
        aux.waist = waist
    basis = np.empty((N+1, N+1, Beam.Dy, Beam.Dx), dtype = 'complex')
    for n in range(N+1):
        for m in range(N+1):
            l = m-n
            p = min(n,m)
            aux.lg(l, p)
            basis[n, m] = aux.field
    if norm1 == True:
        basis = basis/np.max(basis)
    return basis

def bessel_basis(Beam, Nmax, waist=None, norm1=False):
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

def build_from_coefs_and_basis(Beam, coefs, basis):
    N = len(coefs)
    aux = Beam.copy_clean()
    for i in range(N):
        for j in range(N):
            aux.field = aux.field + coefs[i,j] * basis[i,j]
    Beam.field = aux.field
    return Beam

def astigmatic_mode_converter(Beam, N, theta): 
    n, m, overlaps = Beam.hg_projector(N)
    converter = np.exp(1j*(m-n)*theta)
    overlaps = overlaps*converter
    Beam.build_from_coefs_and_basis(overlaps, Beam.hg_basis(N))
    return Beam
