"""Spatial modes: normalization, orthogonality, symmetries and smoke tests."""
import numpy as np
import pytest

from structured_optics import Beam, overlap, hg, lg


def _fresh(beam_factory, **kw):
    return beam_factory(**kw)


HG_INDICES = [(0, 0), (1, 0), (0, 1), (2, 1), (3, 2)]
LG_INDICES = [(0, 0), (1, 0), (-1, 0), (2, 1), (-3, 1)]


@pytest.mark.parametrize("n,m", HG_INDICES)
def test_hg_is_normalized(beam_factory, n, m):
    b = beam_factory(); b.hg(n, m)
    assert b.Power() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("l,p", LG_INDICES)
def test_lg_is_normalized(beam_factory, l, p):
    b = beam_factory(); b.lg(l, p)
    assert b.Power() == pytest.approx(1.0, abs=1e-6)


def test_hg_orthonormality(beam_factory):
    indices = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
    modes = []
    for n, m in indices:
        b = beam_factory(); b.hg(n, m); modes.append(b)
    for i, a in enumerate(modes):
        for j, c in enumerate(modes):
            expected = 1.0 if i == j else 0.0
            assert abs(overlap(a, c)) == pytest.approx(expected, abs=1e-9)


def test_lg_orthonormality(beam_factory):
    indices = [(0, 0), (1, 0), (-1, 0), (2, 1), (0, 2)]
    modes = []
    for l, p in indices:
        b = beam_factory(); b.lg(l, p); modes.append(b)
    for i, a in enumerate(modes):
        for j, c in enumerate(modes):
            expected = 1.0 if i == j else 0.0
            assert abs(overlap(a, c)) == pytest.approx(expected, abs=1e-9)


def test_lg00_equals_hg00(beam_factory):
    a = beam_factory(); a.lg(0, 0)
    b = beam_factory(); b.hg(0, 0)
    assert np.allclose(a.field, b.field, rtol=0, atol=1e-9 * abs(b.field).max())


@pytest.mark.parametrize("n,m", [(1, 0), (2, 0), (1, 1), (3, 2)])
def test_hg_parity(beam_factory, n, m):
    """HG_nm has parity (-1)^n along x and (-1)^m along y."""
    b = beam_factory(); b.hg(n, m)
    assert np.allclose(b.field[:, :, ::-1], (-1) ** n * b.field, atol=1e-9 * abs(b.field).max())
    assert np.allclose(b.field[:, ::-1, :], (-1) ** m * b.field, atol=1e-9 * abs(b.field).max())


def test_angle_rotates_the_mode(beam_factory):
    rotated = beam_factory(); rotated.hg(1, 0, angle=np.pi / 2)
    reference = beam_factory(); reference.hg(0, 1)
    assert np.allclose(rotated.field, reference.field, atol=1e-9 * abs(reference.field).max())


@pytest.mark.parametrize("l", [1, 2, 3, -2])
def test_lg_phase_winds_with_azimuth(beam_factory, l):
    """LG_l,0 carries exp(+i*l*theta): removing it leaves a constant phase."""
    b = beam_factory(Dx=129); b.lg(l, 0)
    x, y = np.broadcast_arrays(b.x, b.y)
    theta = np.arctan2(y, x)
    bright = abs(b.field[0]) > 0.3 * abs(b.field).max()
    residual = b.field[0][bright] * np.exp(-1j * l * theta[bright])
    assert np.ptp(np.angle(residual / residual[0])) < 1e-6


# ------------------------------------------------------------- apertures --

@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_binary_modes_are_normalized_and_binary(beam_factory, shape):
    b = beam_factory(); getattr(b, shape)()
    assert b.Power() == pytest.approx(1.0)
    assert len(np.unique(np.abs(b.field))) == 2  # zero outside, one constant inside


def test_circle_area(beam_factory):
    b = beam_factory(Dx=256); b.circle(radius=2e-3)
    dx = 2 * b.nix / (b.Dx - 1)
    area = (abs(b.field[0]) > 0).sum() * dx * dx
    assert area == pytest.approx(np.pi * (2e-3) ** 2, rel=0.03)


# -------------------------------------------------------------- smoke ---

SMOKE = {
    "bessel": lambda b: b.bessel(3),
    "gbessel": lambda b: b.gbessel(3, 1e-3),
    "lg_prod": lambda b: b.lg_prod(3),
    "frac_oam": lambda b: b.frac_oam(1.5, 5),
    "frac_oam_qs": lambda b: b.frac_oam_qs(1.5, 5),
    "hg_astigmatic": lambda b: b.hg_astigmatic(0, 0, 1e-3, 2e-3),
    "IG_even": lambda b: b.IG_even(2, 0, 1.0),
    "IG_odd": lambda b: b.IG_odd(3, 1, 1.0),
}


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_mode_generators_produce_finite_nonzero_fields(beam_factory, name):
    b = beam_factory(Dx=96)
    returned = SMOKE[name](b)
    assert returned is b  # generators fill the beam in place and return it
    assert np.isfinite(b.field).all() and b.field.any()


def test_helical_ince_gaussian_is_vector_ready(beam_factory):
    b = beam_factory(pol_dim=2, Dx=96); b.HelIG(3, 1, 1.0)
    assert np.isfinite(b.field).all() and b.field.any()


def test_standalone_functions_return_arrays(beam_factory):
    b = beam_factory(Dx=64)
    for field in (hg(b, 1, 0), lg(b, 1, 0)):
        assert field.shape == (b.Dy, b.Dx) and np.isfinite(field).all()


# ------------------------------------------------------------ fiber LP ---

def _fiber_beam(**kw):
    return Beam(nix=20e-6, Dx=128, waist=4e-6, lamb=1.064e-6, **kw)  # waist = core radius


@pytest.mark.parametrize("l,m,parity", [(0, 1, "cos"), (1, 1, "cos"), (1, 1, "sin")])
def test_lp_modes(l, m, parity):
    b = _fiber_beam(); b.lp(l, m, 1.45, 1.44, parity=parity)
    assert np.isfinite(b.field).all() and b.field.any()


def test_lp_hel_two_components():
    b = _fiber_beam(pol_dim=2); b.lp_hel(1, 1, 1.45, 1.44)
    assert np.isfinite(b.field).all() and b.field.any()


def test_lp_mode_that_does_not_propagate_raises():
    with pytest.raises(ValueError):
        _fiber_beam().lp(3, 3, 1.45, 1.44)


@pytest.mark.parametrize("Dx", [48, 96, 255])
def test_normalization_does_not_depend_on_resolution(beam_factory, Dx):
    b = beam_factory(Dx=Dx); b.hg(1, 2)
    assert b.Power() == pytest.approx(1.0, abs=1e-6)
