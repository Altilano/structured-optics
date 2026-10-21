"""Shared fixtures for the structured_optics test-suite.

The package is imported the way a user would (``from structured_optics import ...``),
so run the tests against an installed copy: ``pip install -e ".[test]"``.
"""
import numpy as np
import pytest

from structured_optics import Beam

LAMBDA = 633e-9


def _make_beam(pol_dim=1, Dx=128, Dy=None, nix=6e-3, niy=None, waist=1e-3, lamb=LAMBDA):
    return Beam(nix=nix, Dx=Dx, niy=niy, Dy=Dy, waist=waist, lamb=lamb, pol_dim=pol_dim)


@pytest.fixture
def beam_factory():
    """Callable returning an empty ``Beam``; every argument has a sensible default."""
    return _make_beam


@pytest.fixture
def gaussian(beam_factory):
    """Scalar HG00 beam on a 128x128 grid (waist 1 mm, window 12 mm)."""
    beam = beam_factory()
    beam.hg(0, 0)
    return beam


@pytest.fixture
def random_beam_factory(beam_factory):
    """Callable returning a non-square beam filled with reproducible random complex noise."""

    def _make(pol_dim=2, Dx=61, Dy=48, nix=5e-3, niy=4e-3, seed=0):
        beam = beam_factory(pol_dim=pol_dim, Dx=Dx, Dy=Dy, nix=nix, niy=niy)
        rng = np.random.default_rng(seed)
        shape = beam.field.shape
        beam.field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return beam

    return _make
