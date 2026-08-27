import numpy as np


HPROJ = np.array([[1,0],[0,0]], dtype='complex')
VPROJ = np.array([[0,0],[0,1]], dtype='complex')
DPROJ = np.array([[1,1],[1,1]], dtype='complex')/2
APROJ = np.array([[1,-1],[-1,1]], dtype='complex')/2
RPROJ = np.array([[1,1j],[-1j,1]], dtype='complex')/2
LPROJ = np.array([[1,-1j],[1j,1]], dtype='complex')/2


def J_rot(matrix, ang):
    rt = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]], dtype='complex')
    return rt@matrix@np.conjugate(rt.T)

def J_phase_retarder(delta):
    return np.array([[np.exp(1j*delta/2), 0],[0, np.exp(-1j*delta/2)]])

def J_hwp(ang):
    return J_rot(J_phase_retarder(np.pi/2), ang)

def J_qwp(ang):
    return J_rot(J_phase_retarder(np.pi/4), ang)


