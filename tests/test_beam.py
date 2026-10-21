"""Beam construction, polarization storage, copying and arithmetic."""
import numpy as np
import pytest

from structured_optics import Beam


# ---------------------------------------------------------------- grids --

def test_square_defaults(beam_factory):
    b = beam_factory(Dx=32)
    assert (b.Dx, b.Dy) == (32, 32)
    assert (b.nix, b.niy) == (6e-3, 6e-3)
    assert b.field.shape == (1, 32, 32)
    assert b.field.dtype == np.complex128
    assert not b.field.any()


def test_rectangular_grid(beam_factory):
    b = beam_factory(Dx=40, Dy=24, nix=3e-3, niy=1e-3)
    assert b.field.shape == (1, 24, 40)
    assert b.x.shape == (1, 40) and b.y.shape == (24, 1)  # sparse meshgrid
    assert b.x.min() == pytest.approx(-3e-3) and b.x.max() == pytest.approx(3e-3)
    assert b.y.min() == pytest.approx(-1e-3) and b.y.max() == pytest.approx(1e-3)


@pytest.mark.parametrize("pol_dim", [1, 2, 3])
def test_polarization_dimension(beam_factory, pol_dim):
    b = beam_factory(pol_dim=pol_dim, Dx=16)
    assert b.pol == pol_dim
    assert b.field.shape == (pol_dim, 16, 16)


def test_component_setters_write_into_field(beam_factory):
    b = beam_factory(pol_dim=3, Dx=8)
    b.Ex = np.full((8, 8), 1.0)
    b.Ey = np.full((8, 8), 2.0j)
    b.Ez = np.full((8, 8), 3.0)
    assert np.all(b.field[0] == 1) and np.all(b.field[1] == 2j) and np.all(b.field[2] == 3)
    assert np.shares_memory(b.Ex, b.field)  # property returns a view


# -------------------------------------------------------------- copying --

def test_copy_is_deep(random_beam_factory):
    b = random_beam_factory()
    c = b.copy()
    assert np.array_equal(c.field, b.field)
    c.field[:] = 0
    c.waist = 123.0
    assert b.field.any() and b.waist != 123.0


def test_copy_clean_zeroes_field_but_keeps_grid(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    c = b.copy_clean()
    assert c.field.shape == b.field.shape and not c.field.any()
    assert (c.nix, c.niy, c.Dx, c.Dy, c.lamb, c.waist) == (b.nix, b.niy, b.Dx, b.Dy, b.lamb, b.waist)
    assert b.field.any()  # original untouched
    assert b.copy_clean(pol_dim=3).field.shape == (3, b.Dy, b.Dx)


# ----------------------------------------------------------- arithmetic --

def test_scalar_multiplication_both_sides(random_beam_factory):
    b = random_beam_factory()
    assert np.allclose((2.5 * b).field, 2.5 * b.field)
    assert np.allclose((b * (1 + 1j)).field, (1 + 1j) * b.field)


def test_beam_beam_operations(random_beam_factory):
    a, b = random_beam_factory(seed=1), random_beam_factory(seed=2)
    assert np.allclose((a + b).field, a.field + b.field)
    assert np.allclose((a - b).field, a.field - b.field)
    assert np.allclose((a * b).field, a.field * b.field)


def test_division_by_scalar(random_beam_factory):
    b = random_beam_factory()
    assert np.allclose((b / 4).field, b.field / 4)


def test_arithmetic_returns_new_beams_and_leaves_operands_alone(random_beam_factory):
    a, b = random_beam_factory(seed=1), random_beam_factory(seed=2)
    a0, b0 = a.field.copy(), b.field.copy()
    for result in (a + b, a - b, a * b, 2 * a, a / 2):
        assert result is not a and result is not b
    assert np.array_equal(a.field, a0) and np.array_equal(b.field, b0)


def test_unsupported_operands_raise(random_beam_factory):
    b = random_beam_factory()
    with pytest.raises(TypeError):
        b * "text"
    with pytest.raises(TypeError):
        b / b
    with pytest.raises(TypeError):
        b + 1


# --------------------------------------------------------- polarization --

def test_set_mode_default_is_first_component(beam_factory):
    b = beam_factory(pol_dim=2, Dx=16)
    b._set_mode(np.ones((16, 16)))
    assert np.all(b.field[0] == 1) and not b.field[1].any()


def test_set_mode_normalizes_polarization_vector(beam_factory):
    b = beam_factory(pol_dim=2, Dx=16)
    b._set_mode(np.ones((16, 16)), polarization=[3, 4j])
    assert np.allclose(b.field[0], 0.6) and np.allclose(b.field[1], 0.8j)


def test_set_mode_rejects_bad_polarization(beam_factory):
    b = beam_factory(pol_dim=2, Dx=16)
    with pytest.raises(ValueError):
        b._set_mode(np.ones((16, 16)), polarization=[1, 0, 0])  # wrong length
    with pytest.raises(ValueError):
        b._set_mode(np.ones((16, 16)), polarization=[0, 0])  # zero vector


def test_circular_polarization_from_mode_generator(beam_factory):
    b = beam_factory(pol_dim=2)
    b.hg(0, 0, polarization=[1, 1j])
    centre = (slice(None), b.Dy // 2, b.Dx // 2)
    assert b.field[centre][1] / b.field[centre][0] == pytest.approx(1j)
    assert np.allclose(b.field[1], 1j * b.field[0])


def test_grid_spacing_matches_the_coordinate_arrays(beam_factory):
    b = beam_factory(Dx=40, Dy=24, nix=3e-3, niy=1e-3)
    assert b.dx == pytest.approx(6e-3 / 39) and b.dy == pytest.approx(2e-3 / 23)
    assert b.dx == pytest.approx(b.x[0, 1] - b.x[0, 0])
    assert b.dy == pytest.approx(b.y[1, 0] - b.y[0, 0])
    assert np.allclose(np.diff(b.x[0]), b.dx) and np.allclose(np.diff(b.y[:, 0]), b.dy)


def test_power_is_a_riemann_sum_over_the_true_cell_area(beam_factory):
    b = beam_factory(Dx=40, Dy=24, nix=3e-3, niy=1e-3)
    b.field[:] = 1
    assert b.Power() == pytest.approx(40 * 24 * b.dx * b.dy)
