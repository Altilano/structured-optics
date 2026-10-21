"""SLM and DMD hologram encoders."""
import numpy as np
import pytest

from structured_optics.hologram import dmd_hologram, slm_hologram

SLM_METHODS = ["simple", "g_holo", "davis", "bolduc", "bessel0", "bessel1"]


@pytest.fixture
def target(beam_factory):
    b = beam_factory(nix=3e-3, Dx=96, waist=1e-3); b.lg(1, 0)
    return b


@pytest.mark.parametrize("method", SLM_METHODS)
def test_slm_hologram_is_a_full_range_8bit_image(target, method):
    h = slm_hologram(target, 3, 4, method=method)
    assert h.shape == (target.Dy, target.Dx) and h.dtype == np.uint8
    assert h.min() == 0 and h.max() == 255


def test_slm_hologram_is_deterministic(target):
    assert np.array_equal(slm_hologram(target, 3, 4), slm_hologram(target, 3, 4))


def test_slm_hologram_respects_max_range(target):
    assert slm_hologram(target, 3, 4, max_range=100).max() <= 100


def test_slm_hologram_with_explicit_input_beam(beam_factory, target):
    illumination = beam_factory(nix=3e-3, Dx=96, waist=1.5e-3); illumination.hg(0, 0)
    h = slm_hologram(target, 3, 4, input_beam=illumination)
    assert h.shape == (96, 96)


def test_slm_hologram_depends_on_the_carrier(target):
    assert not np.array_equal(slm_hologram(target, 3, 4), slm_hologram(target, 6, 4))


def test_slm_hologram_rejects_unknown_method(target):
    with pytest.raises(ValueError, match="method must be"):
        slm_hologram(target, 3, 4, method="magic")


def test_dmd_hologram_is_binary_and_signs_are_complementary(target):
    plus = dmd_hologram(target, 0.2, 0.3, sign=1)
    minus = dmd_hologram(target, 0.2, 0.3, sign=-1)
    assert plus.shape == (target.Dy, target.Dx)
    assert set(np.unique(plus)) <= {0.0, 1.0}
    assert np.array_equal(plus + minus, np.ones_like(plus))


def test_dmd_hologram_with_input_beam(beam_factory, target):
    illumination = beam_factory(nix=3e-3, Dx=96, waist=1.5e-3); illumination.hg(0, 0)
    h = dmd_hologram(target, 0.2, 0.3, input=illumination)
    assert h.shape == (96, 96) and set(np.unique(h)) <= {0.0, 1.0}
