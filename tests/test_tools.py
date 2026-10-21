"""Overlaps, mode projections and basis construction."""
import numpy as np
import pytest

from structured_optics import (
    bessel_basis, hg_basis, hg_proj, int_overlap, lg_basis, lg_proj, overlap,
)


def _hg(beam_factory, n, m, **kw):
    b = beam_factory(Dx=96, **kw); b.hg(n, m); return b


def test_overlap_of_a_beam_with_itself_is_one(beam_factory):
    b = _hg(beam_factory, 1, 0)
    assert overlap(b, b) == pytest.approx(1.0)
    assert (3j * b).field.any()
    assert abs(overlap(3j * b, b)) == pytest.approx(1.0)  # normalized: amplitude does not matter


def test_overlap_is_conjugate_symmetric_and_picks_up_the_relative_phase(beam_factory):
    a, b = _hg(beam_factory, 0, 0), _hg(beam_factory, 0, 0)
    assert overlap(a * np.exp(0.7j), b) == pytest.approx(np.exp(0.7j))
    assert overlap(a, b * np.exp(0.7j)) == pytest.approx(np.exp(-0.7j))


def test_orthogonal_modes_have_zero_overlap(beam_factory):
    assert abs(overlap(_hg(beam_factory, 1, 0), _hg(beam_factory, 0, 1))) < 1e-9


def test_intensity_overlap_is_symmetric_and_phase_blind(beam_factory):
    a, b = _hg(beam_factory, 1, 0), _hg(beam_factory, 0, 0)
    assert int_overlap(a, b) == pytest.approx(int_overlap(b, a))
    assert int_overlap(a * np.exp(1.1j), b) == pytest.approx(int_overlap(a, b))
    assert int_overlap(a, b) >= 0


def test_hg_projection_recovers_a_superposition(beam_factory):
    mix = _hg(beam_factory, 1, 0) + 1j * _hg(beam_factory, 0, 2)
    n, m, ov = hg_proj(mix, 3)
    assert n.shape == m.shape == ov.shape == (4, 4)
    assert abs(ov[1, 0]) == pytest.approx(1 / np.sqrt(2), abs=1e-9)
    assert abs(ov[0, 2]) == pytest.approx(1 / np.sqrt(2), abs=1e-9)
    assert ov[0, 2] / ov[1, 0] == pytest.approx(1j)
    others = np.ones((4, 4), bool); others[1, 0] = others[0, 2] = False
    assert np.abs(ov[others]).max() < 1e-9


def test_lg_projection_finds_the_generating_mode(beam_factory):
    b = beam_factory(Dx=96); b.lg(2, 1)
    l, p, ov = lg_proj(b, 3)
    assert ov.shape == (7, 2)
    peak = np.unravel_index(np.abs(ov).argmax(), ov.shape)
    assert (l[peak], p[peak]) == (2, 1)
    assert abs(ov[peak]) == pytest.approx(1.0, abs=1e-6)


def test_projection_does_not_modify_the_beam(beam_factory):
    b = _hg(beam_factory, 1, 1); before = b.field.copy()
    hg_proj(b, 2); lg_proj(b, 2)
    assert np.array_equal(b.field, before)


def test_hg_basis_is_orthonormal(beam_factory):
    b = beam_factory(Dx=96)
    basis = hg_basis(b, 2)
    assert basis.shape == (3, 3, b.Dy, b.Dx)
    flat = basis.reshape(9, -1)
    gram = flat.conj() @ flat.T
    gram = gram / np.sqrt(np.outer(np.diag(gram).real, np.diag(gram).real))
    assert np.allclose(gram, np.eye(9), atol=1e-6)


def test_basis_shapes_and_normalization_flag(beam_factory):
    b = beam_factory(Dx=64)
    assert lg_basis(b, 2).shape[-2:] == (b.Dy, b.Dx)
    assert bessel_basis(b, 2).shape[0] == 5 and bessel_basis(b, 2).shape[-2:] == (b.Dy, b.Dx)
    assert np.max(hg_basis(b, 1, norm1=True)) == pytest.approx(1.0)
    assert hg_basis(b, 1, waist=5e-4).shape == hg_basis(b, 1).shape


def test_build_from_coefficients_reconstructs_modes(beam_factory):
    b = beam_factory(Dx=96)
    basis = hg_basis(b, 2)
    coefs = np.zeros((3, 3), complex); coefs[1, 0] = 1
    rebuilt = beam_factory(Dx=96).build_from_coefs_and_basis(coefs, basis)
    assert np.allclose(rebuilt.Ex, _hg(beam_factory, 1, 0).Ex)

    coefs[:] = 0; coefs[0, 0], coefs[2, 1] = 0.6, 0.8j
    combo = beam_factory(Dx=96).build_from_coefs_and_basis(coefs, basis)
    assert np.allclose(combo.Ex, 0.6 * basis[0, 0] + 0.8j * basis[2, 1])


def test_build_from_coefficients_pol_index(beam_factory):
    b = beam_factory(pol_dim=2, Dx=48)
    basis = hg_basis(b, 1)
    coefs = np.ones((2, 2), complex)
    only_ey = b.copy().build_from_coefs_and_basis(coefs, basis, pol_index=1)
    assert not only_ey.Ex.any() and only_ey.Ey.any()
    every = b.copy().build_from_coefs_and_basis(coefs, basis, pol_index=None)
    assert np.array_equal(every.Ex, every.Ey) and every.Ex.any()
