"""
Polarization optics: Jones-calculus building blocks and Mask subclasses
that act on a Beam's (Ex, Ey) transverse field components.

Jones matrices here follow the convention

    [Ex']   [J00 J01] [Ex]
    [Ey'] = [J10 J11] [Ey]

`JonesMask` wraps a 2x2 Jones matrix as a `Mask` (see `structured_optics.masks`)
so it can be applied to a `Beam` the same way spatial masks are, e.g.:

    beam = beam * Polarizer(angle=0, proj="H") * QWP(angle=np.pi/4)
"""
from abc import abstractmethod
import numpy as np

from structured_optics.masks import Mask

__all__ = [ "HPROJ", "VPROJ", "DPROJ", "APROJ", "RPROJ", "LPROJ",
            "J_rot", "J_phase_retarder", "J_hwp", "J_qwp",
            "HWP", "QWP", "Polarizer"]


HPROJ = np.array([[1,0],[0,0]], dtype='complex')
"""np.ndarray: Jones projector onto horizontal linear polarization."""

VPROJ = np.array([[0,0],[0,1]], dtype='complex')
"""np.ndarray: Jones projector onto vertical linear polarization."""

DPROJ = np.array([[1,1],[1,1]], dtype='complex')/2
"""np.ndarray: Jones projector onto diagonal (+45 deg) linear polarization."""

APROJ = np.array([[1,-1],[-1,1]], dtype='complex')/2
"""np.ndarray: Jones projector onto anti-diagonal (-45 deg) linear polarization."""

RPROJ = np.array([[1,1j],[-1j,1]], dtype='complex')/2
"""np.ndarray: Jones projector onto right-circular polarization."""

LPROJ = np.array([[1,-1j],[1j,1]], dtype='complex')/2
"""np.ndarray: Jones projector onto left-circular polarization."""


def J_rot(matrix, ang):
    """Rotate a Jones matrix about the optical axis by `ang`.

    Applies the similarity transform ``R(ang) @ matrix @ R(ang)^H``, where
    `R(ang)` is the standard 2x2 rotation matrix and `^H` denotes the
    conjugate transpose. This is the general way to express any Jones
    element (retarder, projector, etc.) at an arbitrary orientation.

    Parameters
    ----------
    matrix : np.ndarray
        2x2 Jones matrix to rotate, expressed in the element's own
        (unrotated) basis.
    ang : float
        Rotation angle, in radians.

    Returns
    -------
    np.ndarray
        The rotated 2x2 Jones matrix.
    """
    rt = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]], dtype='complex')
    return rt@matrix@np.conjugate(rt.T)

def J_phase_retarder(delta):
    """Build the Jones matrix of a linear phase retarder in its own basis.

    Parameters
    ----------
    delta : float
        Retardance between the fast and slow axes, in radians.

    Returns
    -------
    np.ndarray
        The 2x2 diagonal Jones matrix
        ``diag(exp(1j*delta/2), exp(-1j*delta/2))``.
    """
    return np.array([[np.exp(1j*delta/2), 0],[0, np.exp(-1j*delta/2)]])

def J_hwp(ang):
    """Build the Jones matrix of a half-wave-plate-type retarder at angle `ang`.

    Parameters
    ----------
    ang : float
        Orientation of the retarder's fast axis, in radians.

    Returns
    -------
    np.ndarray
        The 2x2 Jones matrix from rotating `J_phase_retarder(np.pi/2)` by
        `ang` via `J_rot`.
    """
    return J_rot(J_phase_retarder(np.pi/2), ang)

def J_qwp(ang):
    """Build the Jones matrix of a quarter-wave-plate-type retarder at angle `ang`.

    Parameters
    ----------
    ang : float
        Orientation of the retarder's fast axis, in radians.

    Returns
    -------
    np.ndarray
        The 2x2 Jones matrix from rotating `J_phase_retarder(np.pi/4)` by
        `ang` via `J_rot`.
    """
    return J_rot(J_phase_retarder(np.pi/4), ang)


_PROJECTORS = {"H": HPROJ, "V": VPROJ, "D": DPROJ, "A": APROJ, "R": RPROJ, "L": LPROJ}
"""dict: Lookup from a single-letter polarization code to its Jones
projector matrix. Keys: "H" (horizontal), "V" (vertical), "D" (diagonal),
"A" (anti-diagonal), "R" (right-circular), "L" (left-circular)."""


class JonesMask(Mask):
    """A Mask defined by a 2x2 Jones matrix, acting on (Ex, Ey) only.

    Subclasses implement `matrix(beam)`; Ez (if present) is left untouched.
    """

    @abstractmethod
    def matrix(self, beam) -> np.ndarray:
        """Return the 2x2 Jones matrix to apply to `beam`'s (Ex, Ey).

        Parameters
        ----------
        beam : Beam
            The beam the matrix will be applied to. Provided so
            subclasses can compute a matrix that depends on beam state.

        Returns
        -------
        np.ndarray
            A 2x2 (possibly complex) Jones matrix.
        """
        raise NotImplementedError

    def apply(self, beam):
        """Apply this Jones matrix to a copy of `beam`'s Ex/Ey components.

        Parameters
        ----------
        beam : Beam
            The beam to transform. If `beam.pol == 1` (scalar/non-vector
            beam), the beam is returned unchanged, since there is no
            (Ex, Ey) decomposition to act on.

        Returns
        -------
        Beam
            `beam` unchanged if `beam.pol == 1`; otherwise a copy of
            `beam` with `Ex`, `Ey` replaced by
            ``J @ [Ex, Ey]``, where `J = self.matrix(beam)`. `Ez` (if
            present) and the original `beam` are left untouched.
        """
        if beam.pol == 1:
            return beam
        J = self.matrix(beam)
        result = beam.copy()
        Ex, Ey = beam.Ex.copy(), beam.Ey.copy()
        result.Ex = J[0, 0] * Ex + J[0, 1] * Ey
        result.Ey = J[1, 0] * Ex + J[1, 1] * Ey
        return result

class HWP(JonesMask):
    """A half-wave plate, oriented with its fast axis at `angle`."""

    def __init__(self, angle: float):
        """
        Parameters
        ----------
        angle : float
            Orientation of the fast axis, in radians.
        """
        self.angle = angle

    def matrix(self, beam):
        """Return the half-wave plate's Jones matrix.

        Parameters
        ----------
        beam : Beam
            Unused; accepted to satisfy the `JonesMask.matrix` interface.

        Returns
        -------
        np.ndarray
            The result of ``J_hwp(self.angle)``.
        """
        return J_hwp(self.angle)


class QWP(JonesMask):
    """A quarter-wave plate, oriented with its fast axis at `angle`."""

    def __init__(self, angle: float):
        """
        Parameters
        ----------
        angle : float
            Orientation of the fast axis, in radians.
        """
        self.angle = angle

    def matrix(self, beam):
        """Return the quarter-wave plate's Jones matrix.

        Parameters
        ----------
        beam : Beam
            Unused; accepted to satisfy the `JonesMask.matrix` interface.

        Returns
        -------
        np.ndarray
            The result of ``J_qwp(self.angle)``.
        """
        return J_qwp(self.angle)


class Polarizer(JonesMask):
    """An ideal linear or circular polarizer, oriented at `angle` and
    projecting onto the polarization state named by `proj`."""

    def __init__(self, angle: float = 0, proj: str = "H"):
        """
        Parameters
        ----------
        angle : float, optional
            Rotation applied to the base projector, in radians.
            Defaults to 0.
        proj : str, optional
            Polarization state to project onto before rotation. One of
            "H", "V", "D", "A", "R", "L" (see `_PROJECTORS`). Defaults
            to "H".
        """
        self.angle, self.proj = angle, proj

    def matrix(self, beam):
        """Return the polarizer's Jones matrix.

        Parameters
        ----------
        beam : Beam
            Unused; accepted to satisfy the `JonesMask.matrix` interface.

        Returns
        -------
        np.ndarray
            The result of rotating `_PROJECTORS[self.proj]` by
            `self.angle` via `J_rot`.
        """
        return J_rot(_PROJECTORS[self.proj], self.angle)