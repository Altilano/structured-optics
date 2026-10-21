"""Jones calculus building blocks and polarization masks."""
import numpy as np
import pytest

from structured_optics import HWP, QWP, J_hwp, J_qwp, J_rot, Polarizer
from structured_optics.polarization import (
    APROJ, DPROJ, HPROJ, LPROJ, RPROJ, VPROJ, J_phase_retarder,
)

PROJECTORS = [HPROJ, VPROJ, DPROJ, APROJ, RPROJ, LPROJ]
H = np.array([1, 0], dtype=complex)
V = np.array([0, 1], dtype=complex)
D = np.array([1, 1], dtype=complex) / np.sqrt(2)
A = np.array([1, -1], dtype=complex) / np.sqrt(2)


def same_up_to_global_phase(u, v, atol=1e-9):
    u, v = np.asarray(u).ravel(), np.asarray(v).ravel()
    return abs(abs(np.vdot(u, v)) - np.linalg.norm(u) * np.linalg.norm(v)) < atol


# ------------------------------------------------------------ matrices --

@pytest.mark.parametrize("P", PROJECTORS)
def test_projectors_are_hermitian_idempotent_rank_one(P):
    assert np.allclose(P, P.conj().T)
    assert np.allclose(P @ P, P)
    assert np.trace(P).real == pytest.approx(1.0)


@pytest.mark.parametrize("P,Q", [(HPROJ, VPROJ), (DPROJ, APROJ), (RPROJ, LPROJ)])
def test_orthogonal_projector_pairs_sum_to_identity(P, Q):
    assert np.allclose(P + Q, np.eye(2))
    assert np.allclose(P @ Q, 0)


def test_rotation_of_projectors():
    assert np.allclose(J_rot(HPROJ, 0), HPROJ)
    assert np.allclose(J_rot(HPROJ, np.pi / 2), VPROJ)
    assert np.allclose(J_rot(HPROJ, np.pi / 4), DPROJ)


def test_rotations_compose():
    M = J_phase_retarder(0.7)
    assert np.allclose(J_rot(J_rot(M, 0.3), 0.5), J_rot(M, 0.8))


@pytest.mark.parametrize("make", [J_phase_retarder, lambda a: J_hwp(a), lambda a: J_qwp(a)])
@pytest.mark.parametrize("arg", [0.0, 0.4, np.pi / 3])
def test_wave_plates_are_unitary(make, arg):
    J = make(arg)
    assert np.allclose(J @ J.conj().T, np.eye(2))


# ---------------------------------------------------------------- masks --

def test_polarizer_projects_a_vector_beam(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    out = b * Polarizer(proj="H")
    assert np.array_equal(out.field[0], b.field[0]) and not out.field[1].any()


def test_crossed_polarizers_block_everything(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    assert not (b * Polarizer(proj="H") * Polarizer(proj="V")).field.any()


@pytest.mark.parametrize("angle", [np.pi / 6, np.pi / 3, 1.2])
def test_malus_law(beam_factory, angle):
    b = beam_factory(pol_dim=2); b.hg(0, 0)  # horizontally polarized
    out = b * Polarizer(angle=angle, proj="H")
    assert out.Power() == pytest.approx(np.cos(angle) ** 2 * b.Power(), rel=1e-12)


def test_polarizer_rejects_unknown_projector_name():
    with pytest.raises(KeyError):
        Polarizer(proj="Z").matrix(None)


def test_jones_masks_leave_ez_and_input_untouched(random_beam_factory):
    b = random_beam_factory(pol_dim=3); orig = b.field.copy()
    for mask in (HWP(0.3), QWP(0.3), Polarizer(0.3, "D")):
        out = b * mask
        assert np.array_equal(out.field[2], orig[2])
        assert np.array_equal(b.field, orig)


def test_jones_masks_are_noops_on_scalar_beams(random_beam_factory):
    b = random_beam_factory(pol_dim=1)
    for mask in (HWP(0.3), QWP(0.3), Polarizer(0.3, "D")):
        assert np.array_equal((b * mask).field, b.field)


def test_mask_classes_use_their_jones_matrices(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    assert np.allclose(HWP(0.3).matrix(b), J_hwp(0.3))
    assert np.allclose(QWP(0.3).matrix(b), J_qwp(0.3))


# ------------------------------------------ physical wave-plate behaviour --


def test_hwp_squares_to_minus_identity():
    assert np.allclose(J_hwp(0) @ J_hwp(0), -np.eye(2))


def test_hwp_swaps_diagonal_polarizations():
    assert same_up_to_global_phase(J_hwp(0) @ D, A)


def test_hwp_at_45_degrees_turns_h_into_v():
    assert same_up_to_global_phase(J_hwp(np.pi / 4) @ H, V)


def test_qwp_turns_diagonal_into_circular():
    out = J_qwp(0) @ D
    assert abs(out[0]) == pytest.approx(abs(out[1]))
    assert abs(np.angle(out[1] / out[0])) == pytest.approx(np.pi / 2)


def test_two_quarter_wave_plates_make_a_half_wave_plate():
    assert same_up_to_global_phase(J_qwp(0.3) @ J_qwp(0.3), J_hwp(0.3))
