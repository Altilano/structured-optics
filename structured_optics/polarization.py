from abc import abstractmethod
import numpy as np

from structured_optics.masks import Mask

__all__ = [ "HPROJ", "VPROJ", "DPROJ", "APROJ", "RPROJ", "LPROJ",
            "J_rot", "J_phase_retarder", "J_hwp", "J_qwp",
            "HWP", "QWP", "Polarizer"]


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


_PROJECTORS = {"H": HPROJ, "V": VPROJ, "D": DPROJ, "A": APROJ, "R": RPROJ, "L": LPROJ}


class JonesMask(Mask):
    """A Mask defined by a 2x2 Jones matrix, acting on (Ex, Ey) only.

    Subclasses implement `matrix(beam)`; Ez (if present) is left untouched.
    """

    @abstractmethod
    def matrix(self, beam) -> np.ndarray:
        raise NotImplementedError

    def apply(self, beam):
        if beam.pol == 1:
            return beam
        J = self.matrix(beam)
        result = beam.copy()
        Ex, Ey = beam.Ex.copy(), beam.Ey.copy()
        result.Ex = J[0, 0] * Ex + J[0, 1] * Ey
        result.Ey = J[1, 0] * Ex + J[1, 1] * Ey
        return result

class HWP(JonesMask):
    def __init__(self, angle: float):
        self.angle = angle

    def matrix(self, beam):
        return J_hwp(self.angle)


class QWP(JonesMask):
    def __init__(self, angle: float):
        self.angle = angle

    def matrix(self, beam):
        return J_qwp(self.angle)


class Polarizer(JonesMask):
    def __init__(self, angle: float = 0, proj: str = "H"):
        self.angle, self.proj = angle, proj

    def matrix(self, beam):
        return J_rot(_PROJECTORS[self.proj], self.angle)