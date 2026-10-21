"""Beam measurements: power, profiles, moments, angular sections, normalization."""
import numpy as np
import pytest

LAMB, W0 = 633e-9, 1e-3


def test_power_of_a_normalized_mode(gaussian):
    assert gaussian.Power() == pytest.approx(1.0, abs=1e-6)


def test_power_adds_over_polarization_components(random_beam_factory):
    b = random_beam_factory(pol_dim=3)
    assert b.Power() == pytest.approx(sum(b.Power(i) for i in range(3)))
    assert np.allclose(b.int_profile(), sum(b.int_profile(i) for i in range(3)))


def test_int_profile_selects_component(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    assert np.allclose(b.int_profile(1), np.abs(b.field[1]) ** 2)
    assert b.int_profile().shape == (b.Dy, b.Dx)


def test_power_scales_quadratically_with_amplitude(gaussian):
    assert (3 * gaussian).Power() == pytest.approx(9 * gaussian.Power())


def test_norm_beam_sets_unit_power_in_place(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    assert b.norm_beam() is b
    assert b.Power() == pytest.approx(1.0, rel=1e-12)


def test_max_int1_sets_unit_peak_intensity(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    assert b.Max_int1() is b
    assert b.int_profile().max() == pytest.approx(1.0, rel=1e-12)


def test_gaussian_moments(gaussian):
    assert gaussian.center_mass() == pytest.approx((0.0, 0.0), abs=1e-12)
    sx, sy = gaussian.std()
    assert sx == pytest.approx(W0 / 2, rel=1e-6) and sy == pytest.approx(W0 / 2, rel=1e-6)
    assert gaussian.radial_std() == pytest.approx(W0 / np.sqrt(2), rel=1e-6)


def test_center_mass_follows_an_off_axis_aperture(beam_factory):
    b = beam_factory(Dx=256); b.circle(center=(1e-3, -2e-3), radius=5e-4)
    cx, cy = b.center_mass()
    pixel = 2 * b.nix / (b.Dx - 1)
    assert cx == pytest.approx(1e-3, abs=pixel) and cy == pytest.approx(-2e-3, abs=pixel)


def test_physical_constants_from_beam_parameters(gaussian):
    assert gaussian.k() == pytest.approx(2 * np.pi / LAMB)
    assert gaussian.zr() == pytest.approx(np.pi * W0 ** 2 / LAMB)


def test_phase_of_a_known_constant_phase(gaussian):
    shifted = gaussian * np.exp(0.5j)
    centre = (gaussian.Dy // 2, gaussian.Dx // 2)
    assert shifted.phase()[centre] == pytest.approx(0.5)
    negative = gaussian * np.exp(-0.5j)
    assert negative.phase()[centre] == pytest.approx(-0.5)
    assert negative.phase(twopi=True)[centre] == pytest.approx(2 * np.pi - 0.5)


# ------------------------------------------------------------- sections --

QUADRANTS = [(k * np.pi / 2, (k + 1) * np.pi / 2) for k in range(4)]


def test_section_powers_add_up_to_total(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    for pol in (0, 1, None):
        total = b.Power(pol)
        parts = sum(b.Power_section(lo, hi, pol_index=pol) for lo, hi in QUADRANTS)
        assert parts == pytest.approx(total, rel=1e-12)


def test_section_intensities_add_up_to_total(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    parts = sum(b.int_section(lo, hi, pol_index=1) for lo, hi in QUADRANTS)
    assert np.allclose(parts, b.int_profile(1))


def test_section_intensity_is_zero_outside_the_wedge(random_beam_factory):
    b = random_beam_factory(pol_dim=2)
    x, y = np.broadcast_arrays(b.x, b.y)
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
    sec = b.int_section(0.4, 1.9, pol_index=0)
    outside = ~((theta >= 0.4) & (theta < 1.9))
    assert not sec[outside].any()
    assert np.allclose(sec[~outside], np.abs(b.field[0][~outside]) ** 2)


def test_symmetric_beam_splits_power_evenly(gaussian):
    upper = gaussian.Power_section(0, np.pi, pol_index=0)
    lower = gaussian.Power_section(np.pi, 2 * np.pi, pol_index=0)
    assert upper == pytest.approx(lower, rel=0.05)
    assert upper + lower == pytest.approx(gaussian.Power(), rel=1e-12)


def test_old_section_method_is_gone(gaussian):
    """Angular sections are now expressed as the AngularSection mask."""
    assert not hasattr(gaussian, "section")
