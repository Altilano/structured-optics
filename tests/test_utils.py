"""Numerical helper functions: special functions, inverses, Zernikes, fiber modes."""
import numpy as np
import pytest
from scipy.special import j0, j1

from structured_optics.utils import (
    characteristic_eq, cartesian_to_elliptic, effective_index, elliptic_to_cartesian,
    find_all_LP_modes, get_LP_params, herm, inv_J0, inv_J1, inv_sinc, laguerre,
    lp_cutoff_V, radial_poly, v_number, zernike,
)

X = np.linspace(-1, 1, 5)
J1_FIRST_MAX = 1.8411837813406593  # location of the first maximum of J1


def test_hermite_polynomials():
    assert np.allclose(herm(X, 0), 1)
    assert np.allclose(herm(X, 1), 2 * X)
    assert np.allclose(herm(X, 2), 4 * X ** 2 - 2)


def test_laguerre_polynomials():
    assert np.allclose(laguerre(X, 0, 1), 1 - X)
    assert np.allclose(laguerre(X, 1, 1), 2 - X)
    assert np.allclose(laguerre(X, 0, 2), 1 - 2 * X + X ** 2 / 2)


# -------------------------------------------------- inverse special functions --

A = np.linspace(0.05, 0.95, 7)


def test_inv_sinc_roundtrip():
    assert np.allclose(np.sinc(inv_sinc(A) / np.pi), A, atol=1e-6)


def test_inv_j0_roundtrip():
    assert np.allclose(j0(inv_J0(A)), A, atol=1e-6)


def test_inv_j1_roundtrip_is_relative_to_the_first_maximum():
    x = inv_J1(A)
    assert np.allclose(j1(x) / j1(J1_FIRST_MAX), A, atol=1e-6)
    assert inv_J1(np.array([1.0]))[0] == pytest.approx(J1_FIRST_MAX, abs=1e-3)
    assert inv_J1(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------- Zernike --

R = np.linspace(0, 1, 9)  # radial coordinate is non-negative


def test_radial_polynomials_with_m_zero():
    assert np.allclose(radial_poly(0, 0, R), 1)
    assert np.allclose(radial_poly(2, 0, R), 2 * R ** 2 - 1)
    assert np.allclose(radial_poly(4, 0, R), 6 * R ** 4 - 6 * R ** 2 + 1)
    assert np.allclose(radial_poly(1, 1, R), R)


def test_radial_polynomials_reject_invalid_orders():
    with pytest.raises(ValueError):
        radial_poly(2, 1, R)   # n - m odd
    with pytest.raises(ValueError):
        radial_poly(1, 3, R)   # m > n


def test_radial_polynomials_with_nonzero_m_match_closed_forms():
    assert np.allclose(radial_poly(2, 2, R), R ** 2)
    assert np.allclose(radial_poly(3, 1, R), 3 * R ** 3 - 2 * R)
    assert np.allclose(radial_poly(3, 3, R), R ** 3)
    assert np.allclose(radial_poly(4, 2, R), 4 * R ** 4 - 3 * R ** 2)


def test_radial_polynomials_equal_one_at_the_edge():
    for n, m in [(2, 2), (3, 1), (3, 3), (4, 2), (4, 4), (5, 1)]:
        assert radial_poly(n, m, np.array([1.0]))[0] == pytest.approx(1.0), (n, m)


def test_zernike_low_orders():
    r, phi = np.array([0.5, 1.0]), np.array([0.3, 0.3])
    assert np.allclose(zernike(0, 0, r, phi), 1)
    assert np.allclose(zernike(1, 1, r, phi), r * np.cos(phi))
    assert np.allclose(zernike(1, -1, r, phi), r * np.sin(phi))
    assert np.allclose(zernike(2, 0, r, 0 * phi), 2 * r ** 2 - 1)


# ---------------------------------------------------------- fiber LP modes --

N_CORE, N_CLAD, RADIUS, LAMBDA = 1.45, 1.44, 4e-6, 1.064e-6


def test_v_number_formula():
    expected = 2 * np.pi * RADIUS / LAMBDA * np.sqrt(N_CORE ** 2 - N_CLAD ** 2)
    assert v_number(N_CORE, N_CLAD, RADIUS, LAMBDA) == pytest.approx(expected)


def test_lp_cutoffs_are_bessel_zeros():
    assert lp_cutoff_V(0, 1) == 0.0
    assert lp_cutoff_V(1, 1) == pytest.approx(2.404825557695773)   # first zero of J0
    assert lp_cutoff_V(2, 1) == pytest.approx(3.831705970207512)   # first zero of J1
    assert lp_cutoff_V(0, 2) == pytest.approx(3.831705970207512)


def test_lp_parameters_solve_the_eigenvalue_equation():
    p = get_LP_params(1, 1, N_CORE, N_CLAD, RADIUS, LAMBDA)
    V = v_number(N_CORE, N_CLAD, RADIUS, LAMBDA)
    assert p["V"] == pytest.approx(V)
    assert p["u"] ** 2 + p["v"] ** 2 == pytest.approx(V ** 2)
    assert abs(characteristic_eq(p["u"], V, 1)) < 1e-8
    assert N_CLAD < p["n_eff"] < N_CORE
    assert p["n_eff"] == pytest.approx(effective_index(p["u"], V, N_CORE, N_CLAD))


def test_lp_modes_below_cutoff_raise():
    with pytest.raises(ValueError, match="does not exist"):
        get_LP_params(3, 3, N_CORE, N_CLAD, RADIUS, LAMBDA)


def test_fundamental_mode_always_propagates():
    p = get_LP_params(0, 1, N_CORE, N_CLAD, 1e-6, LAMBDA)  # even a very thin fiber guides LP01
    assert p["V_cutoff"] == 0.0


def test_find_all_lp_modes_returns_only_guided_modes():
    modes = find_all_LP_modes(3.5, l_max=2, m_max=2)
    assert set(modes) == {(0, 1), (1, 1)}      # LP02 and LP21 need V > 3.83
    for u, v in modes.values():
        assert 0 < u < 3.5 and u ** 2 + v ** 2 == pytest.approx(3.5 ** 2)


# ------------------------------------------------------ elliptic coordinates --

def test_elliptic_coordinates_roundtrip():
    x = np.array([1e-4, 2e-4, -3e-4]); y = np.array([-1e-4, 3e-4, 5e-5])
    xi, eta = cartesian_to_elliptic(x, y, 1.0, 1e-3, 0.0, 633e-9)
    x2, y2 = elliptic_to_cartesian(xi, eta, 1.0, 1e-3, 0.0, 633e-9)
    assert np.allclose(x2, x, rtol=1e-9, atol=0) and np.allclose(y2, y, rtol=1e-9, atol=0)
