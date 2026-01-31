import numpy as np

def std_polh():
    return np.array([1,0], dtype='complex')
    
def norm(v):
    N = np.dot(v, np.conjugate(v))
    v = v/np.sqrt(N)
    return v

def H(v):
    proj = np.array([[1,0],[0,0]], dtype='complex')
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def V(v):
    proj = np.array([[0,0],[0,1]], dtype='complex')
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def D(v):
    proj = np.array([[1,1],[1,1]], dtype='complex')/2
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def A(v):
    proj = np.array([[1,-1],[-1,1]], dtype='complex')/2
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def R(v):
    proj = np.array([[1,-1j],[1j,1]], dtype='complex')/2
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def L(v):
    proj = np.array([[1,1j],[-1j,1]], dtype='complex')/2
    polt = np.transpose(np.conjugate(v))
    return np.real(np.dot(polt,np.matmul(proj, v)))

def rot(v, ang):
    rt = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]], dtype='complex')
    return np.matmul(rt, v)
