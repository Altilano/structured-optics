"""Free-space propagation, Bluestein transforms and method selection."""
import numpy as np
import pytest

from structured_optics.bluestein import fft2_bluestein, fft_bluestein
from structured_optics.prop_methods import estimate_bluestein_range, suggest_propagation_method

W0, LAMB = 1e-3, 633e-9
ZR = np.pi * W0 ** 2 / LAMB


def sigma(z):
    """Analytic 1/e^2-radius/2 of a Gaussian beam: intensity std along one axis."""
    return W0 * np.sqrt(1 + (z / ZR) ** 2) / 2


@pytest.fixture
def waist_beam(beam_factory):
    b = beam_factory(Dx=256, waist=W0, lamb=LAMB); b.hg(0, 0)
    return b


# --------------------------------------------------- free-space methods --

@pytest.mark.parametrize("method", ["fres_c", "AS"])
@pytest.mark.parametrize("z", [0.1, 0.5, 2.0])
def test_gaussian_spreads_like_theory(waist_beam, method, z):
    out = waist_beam.propagate(z, method=method)
    assert out.std()[0] == pytest.approx(sigma(z), rel=1e-4)
    assert out.std()[1] == pytest.approx(sigma(z), rel=1e-4)


@pytest.mark.parametrize("method", ["fres_c", "AS"])
def test_power_is_conserved(waist_beam, method):
    p0 = waist_beam.Power()
    assert waist_beam.copy().propagate(0.7, method=method).Power() == pytest.approx(p0, rel=1e-8)


@pytest.mark.parametrize("method", ["fres_c", "AS"])
def test_forward_then_backward_returns_the_input(waist_beam, method):
    out = waist_beam.copy().propagate(0.6, method=method).propagate(-0.6, method=method)
    assert np.allclose(out.field, waist_beam.field, rtol=0, atol=1e-8 * abs(waist_beam.field).max())


@pytest.mark.parametrize("method", ["fres_c", "AS"])
def test_zero_distance_is_identity(waist_beam, method):
    out = waist_beam.copy().propagate(0, method=method)
    assert np.allclose(out.field, waist_beam.field, rtol=0, atol=1e-9 * abs(waist_beam.field).max())


def test_propagation_is_linear(beam_factory):
    a, b = beam_factory(Dx=128), beam_factory(Dx=128)
    a.hg(1, 0); b.hg(0, 2)
    summed = (a + 2j * b).propagate(0.4, method="AS")
    separate = a.copy().propagate(0.4, method="AS") + 2j * b.copy().propagate(0.4, method="AS")
    assert np.allclose(summed.field, separate.field, rtol=0, atol=1e-9 * abs(summed.field).max())


@pytest.mark.parametrize("method", ["fres_f", "blue_fix"])
def test_resampling_methods_agree_with_theory(waist_beam, method):
    out = waist_beam.propagate(0.5, method=method)
    assert np.isfinite(out.field).all()
    assert out.std()[0] == pytest.approx(sigma(0.5), rel=1e-3)
    assert out.std()[1] == pytest.approx(sigma(0.5), rel=1e-3)


def test_fresnel_fft_conserves_power_and_updates_the_grid_description(waist_beam):
    p0 = waist_beam.Power()
    out = waist_beam.propagate(0.5, method="fres_f")
    assert out.Power() == pytest.approx(p0, rel=1e-8)
    # nix/niy and kx/ky must describe the resampled grid, otherwise Power() and any
    # later propagation use the old pitch.
    assert out.dx == pytest.approx(abs(out.x[0, 1] - out.x[0, 0]))
    assert out.dy == pytest.approx(abs(out.y[1, 0] - out.y[0, 0]))
    assert out.kx[0, 1] == pytest.approx(2 * np.pi / (out.Dx * out.dx))


def test_blue_fix_loses_only_what_falls_outside_the_window(waist_beam):
    p0 = waist_beam.Power()
    assert waist_beam.propagate(0.5, method="blue_fix").Power() == pytest.approx(p0, rel=1e-4)


def test_fraunhofer_rescales_the_grid(beam_factory):
    b = beam_factory(nix=2e-3, Dx=128, waist=1e-4); b.circle(radius=1e-4)
    nix0, p0 = b.nix, b.Power()
    out = b.propagate(2.0, method="fraun")
    assert out.nix > 5 * nix0
    assert out.field.shape == (1, 128, 128) and np.isfinite(out.field).all()
    assert out.Power() == pytest.approx(p0, rel=1e-8)
    centre = abs(out.field[0, 64, 64])
    assert centre == pytest.approx(abs(out.field).max(), rel=0.05)  # bright on-axis Airy peak


def test_incoherent_propagation_returns_finite_amplitude(waist_beam):
    out = waist_beam.propagate(0.3, method="inc")
    assert np.isfinite(out.field).all() and (np.abs(out.field) >= 0).all()


def test_propagate_works_in_place_and_returns_self(waist_beam):
    assert waist_beam.propagate(0.2, method="AS") is waist_beam


def test_propagate_works_on_vector_beams(beam_factory):
    b = beam_factory(pol_dim=2, Dx=128); b.hg(0, 0, polarization=[1, 1j])
    out = b.propagate(0.5, method="AS")
    assert np.allclose(out.field[1], 1j * out.field[0], atol=1e-9 * abs(out.field).max())


def test_unknown_method_lists_valid_options(waist_beam):
    with pytest.raises(ValueError, match="Valid options"):
        waist_beam.propagate(1.0, method="teleport")


def test_auto_method_returns_the_beam(beam_factory):
    b = beam_factory(Dx=64, waist=W0, lamb=LAMB); b.hg(0, 0)
    assert b.propagate(0.3, method="auto") is b


# ---------------------------------------------------- method suggestions --

@pytest.mark.parametrize("z", [1e-3, 0.5, 50.0, 1e4])
def test_suggest_returns_a_valid_method_and_diagnostics(waist_beam, z):
    method, info = suggest_propagation_method(waist_beam, z)
    assert method in {"AS", "fres_c", "fres_f", "fraun", "blue", "blue_fix", "inc", "none"}
    assert isinstance(info, dict)


def test_suggest_extremes(waist_beam):
    assert suggest_propagation_method(waist_beam, 0)[0] == "none"
    assert suggest_propagation_method(waist_beam, 1e4)[0] == "fraun"


def test_bluestein_window_estimate_brackets_the_beam(waist_beam):
    x_range, y_range, info = estimate_bluestein_range(waist_beam, 0.5)
    for lo, hi in (x_range, y_range):
        assert lo < 0 < hi
        assert hi >= 4 * sigma(0.5)  # default window is several standard deviations wide


# ---------------------------------------------------------- bluestein core --

def _direct_ft(f, dx, dk, d_out, x0, k0):
    x = x0 + dx * np.arange(f.shape[-1])
    k = k0 + dk * np.arange(d_out)
    return (f[None, :] * np.exp(-1j * k[:, None] * x[None, :])).sum(axis=1) * dx


def test_bluestein_forward_matches_direct_sum():
    rng = np.random.default_rng(3)
    f = rng.normal(size=37) + 1j * rng.normal(size=37)
    dx, dk, x0, k0, d_out = 0.1, 0.07, -1.3, 0.4, 50
    F, k = fft_bluestein(f, dx, dk, D_out=d_out, x0=x0, k0=k0)
    expected = _direct_ft(f, dx, dk, d_out, x0, k0)
    assert np.allclose(F, expected, rtol=0, atol=1e-10 * abs(expected).max())
    assert np.allclose(k, k0 + dk * np.arange(d_out))


def test_bluestein_inverse_matches_direct_sum():
    rng = np.random.default_rng(4)
    F = rng.normal(size=37) + 1j * rng.normal(size=37)
    dx, dk, x0, k0, d_out = 0.1, 0.07, -1.3, 0.4, 50
    f, x = fft_bluestein(F, dx, dk, D_out=d_out, x0=x0, k0=k0, inverse=True)
    k_in = k0 + dk * np.arange(F.size)
    x_out = x0 + dx * np.arange(d_out)
    expected = (F[None, :] * np.exp(1j * x_out[:, None] * k_in[None, :])).sum(axis=1) * dk / (2 * np.pi)
    assert np.allclose(f, expected, rtol=0, atol=1e-10 * abs(expected).max())
    assert np.allclose(x, x_out)


def test_bluestein_default_output_length_and_batching():
    rng = np.random.default_rng(5)
    f = rng.normal(size=(3, 20)) + 0j
    F, k = fft_bluestein(f, 0.1, 0.2)
    assert F.shape == (3, 20) and k.shape == (20,)


def test_bluestein_2d_matches_direct_sum():
    rng = np.random.default_rng(6)
    field = rng.normal(size=(16, 20)) + 1j * rng.normal(size=(16, 20))
    dx = dy = 0.1; dkx = dky = 0.05
    out, kx, ky = fft2_bluestein(field, dx, dy, dkx, dky, Dx_out=24, Dy_out=12)
    x, y = dx * np.arange(20), dy * np.arange(16)
    kxg, kyg = dkx * np.arange(24), dky * np.arange(12)
    direct = np.einsum("yx,ky,lx->kl", field, np.exp(-1j * kyg[:, None] * y[None, :]),
                       np.exp(-1j * kxg[:, None] * x[None, :])) * dx * dy
    assert out.shape == (12, 24) and kx.shape == (24,) and ky.shape == (12,)
    assert np.allclose(out, direct, rtol=0, atol=1e-10 * abs(direct).max())


def test_bluestein_validates_arguments():
    with pytest.raises(ValueError):
        fft_bluestein(np.ones(8), dx=-1.0, dk=1.0)
    with pytest.raises(ValueError):
        fft_bluestein(np.ones(8), dx=1.0, dk=0.0)
    with pytest.raises(ValueError):
        fft2_bluestein(np.ones(8), 1, 1, 1, 1)  # needs a 2D array
