"""Spatial masks, apertures, AngularSection and restrict_mask."""
import numpy as np
import pytest

from structured_optics import (
    AngularSection, AstigmaticLens, CrossSlit, DoubleSlit, HSlit, Iris, Lens, Polarizer,
    QWP, RegionMask, SquareAperture, TiltedLens, TriangleAperture, VSlit, ZernikeMask,
    restrict_mask,
)
from structured_optics.masks import BooleanMask, Mask


# ------------------------------------------------------------ protocol --

def test_mask_application_does_not_modify_input(random_beam_factory):
    b = random_beam_factory()
    before = b.field.copy()
    out = b * Lens(0.5)
    assert out is not b and np.array_equal(b.field, before)


def test_masks_chain_left_to_right(random_beam_factory):
    b = random_beam_factory(pol_dim=1)
    lens, iris = Lens(0.5), Iris(radius=2e-3)
    assert np.allclose((b * lens * iris).field, iris.apply(lens.apply(b)).field)


def test_mask_rmul_rejects_non_beams():
    with pytest.raises(TypeError):
        2 * Lens(1.0)


def test_custom_mask_subclass_works_with_operator(random_beam_factory):
    class Flip(Mask):
        def apply(self, beam):
            out = beam.copy(); out.field = -out.field; return out

    b = random_beam_factory()
    assert np.allclose((b * Flip()).field, -b.field)


# --------------------------------------------------------------- lenses --

def test_lens_focuses_gaussian_to_analytic_waist(beam_factory):
    """Thin lens at the waist of a Gaussian: check focal-plane size against ABCD optics."""
    w0, lamb, f = 1e-3, 633e-9, 1.0
    b = beam_factory(Dx=512, waist=w0, lamb=lamb); b.hg(0, 0)
    zr = np.pi * w0 ** 2 / lamb
    z_focus = f / (1 + (f / zr) ** 2)
    w_focus = w0 / np.sqrt(1 + (zr / f) ** 2)

    focused = (b * Lens(f)).propagate(z_focus, method="AS")
    assert focused.std()[0] == pytest.approx(w_focus / 2, rel=1e-3)

    diverging = (b * Lens(-f)).propagate(z_focus, method="AS")
    assert diverging.std()[0] > w0 / 2  # negative focal length spreads the beam


def test_lens_is_pure_phase_and_symmetric(beam_factory):
    b = beam_factory(Dx=65)
    arr = Lens(0.3).array(b)
    assert np.allclose(np.abs(arr), 1)
    assert np.allclose(arr, arr[::-1, :]) and np.allclose(arr, arr[:, ::-1])


def test_lens_offset_moves_the_symmetry_centre(beam_factory):
    b = beam_factory(Dx=128)
    kx, ky, j = 70, 40, 10
    arr = Lens(0.3, f0=(b.x[0, kx], b.y[ky, 0])).array(b)
    assert np.allclose(arr[:, kx + j], arr[:, kx - j])
    assert np.allclose(arr[ky + j, :], arr[ky - j, :])


def test_astigmatic_lens_with_equal_focals_is_a_lens(beam_factory):
    b = beam_factory()
    assert np.allclose(AstigmaticLens(0.4, 0.4).array(b), Lens(0.4).array(b))


def test_tilted_lens_reduces_to_lens_at_zero_tilt_and_flips_axes(beam_factory):
    b = beam_factory()
    assert np.allclose(TiltedLens(0.5, phi=0.0).array(b), Lens(0.5).array(b))
    t, flipped = TiltedLens(0.5, phi=0.4), TiltedLens(0.5, phi=0.4, flip_axis=True)
    assert (flipped.fx, flipped.fy) == (t.fy, t.fx)
    assert t.fx < t.fy  # tilt shortens the effective focal length along one axis


def test_tilted_lens_requires_index_above_one():
    with pytest.raises(ValueError):
        TiltedLens(0.5, phi=0.1, n=1.0)


def test_zernike_mask_is_pure_phase_and_identity_at_zero_strength(beam_factory):
    b = beam_factory()
    assert np.allclose(ZernikeMask(((2, 0),), (0.0,)).array(b), 1)
    arr = ZernikeMask(((1, 1), (2, -2)), (0.5, 0.3)).array(b)
    assert arr.shape == (b.Dy, b.Dx)
    assert np.allclose(np.abs(arr), 1) and not np.allclose(arr, 1)


# ----------------------------------------------------------- apertures --

def _regions(beam):
    return {
        "iris": Iris(radius=2e-3, center=(1e-3, -5e-4)),
        "square": SquareAperture(side_length=3e-3),
        "triangle": TriangleAperture(side_length=4e-3),
        "hslit": HSlit(size=1e-3, center=5e-4),
        "vslit": VSlit(size=1e-3, center=-5e-4),
        "double": DoubleSlit(size=3e-4, dis=2e-3),
        "cross": CrossSlit(size=5e-4, hsize=1e-3),
        "angular": AngularSection(0.3, 2.0),
    }


def test_every_boolean_mask_is_boolean_and_nonempty(beam_factory):
    b = beam_factory()
    for name, mask in _regions(b).items():
        assert isinstance(mask, BooleanMask), name
        arr = mask.array(b)
        assert arr.dtype == bool and arr.shape == (b.Dy, b.Dx), name
        assert 0 < arr.sum() < arr.size, name


def test_complementary_is_exact_negation(beam_factory):
    b = beam_factory()
    kinds = {
        "iris": lambda c: Iris(radius=2e-3, complementary=c),
        "square": lambda c: SquareAperture(side_length=3e-3, complementary=c),
        "triangle": lambda c: TriangleAperture(side_length=4e-3, complementary=c),
        "hslit": lambda c: HSlit(size=1e-3, complementary=c),
        "vslit": lambda c: VSlit(size=1e-3, complementary=c),
        "double": lambda c: DoubleSlit(size=3e-4, dis=2e-3, complementary=c),
        "cross": lambda c: CrossSlit(size=5e-4, complementary=c),
        "angular": lambda c: AngularSection(0.3, 2.0, complementary=c),
    }
    for name, make in kinds.items():
        assert np.array_equal(make(False).array(b), ~make(True).array(b)), name


def test_aperture_keeps_field_inside_and_blocks_outside(random_beam_factory):
    b = random_beam_factory()
    mask = Iris(radius=1.5e-3)
    inside = mask.array(b)
    out = b * mask
    assert np.array_equal(out.field[:, inside], b.field[:, inside])
    assert not out.field[:, ~inside].any()


def test_iris_area(beam_factory):
    b = beam_factory(Dx=256)
    dx = 2 * b.nix / (b.Dx - 1)
    area = Iris(radius=2e-3).array(b).sum() * dx * dx
    assert area == pytest.approx(np.pi * (2e-3) ** 2, rel=0.03)


def test_slits_select_the_right_axis(beam_factory):
    b = beam_factory(Dx=65)
    h = HSlit(size=1e-3).array(b)   # |x| <= size -> constant along y
    v = VSlit(size=1e-3).array(b)   # |y| <= size -> constant along x
    assert np.array_equal(h, np.broadcast_to(h[0], h.shape))
    assert np.array_equal(v, np.broadcast_to(v[:, :1], v.shape))
    ds = DoubleSlit(size=2e-4, dis=3e-3).array(b)[0]
    assert not ds[len(ds) // 2]  # blocked on axis between the two slits


# ------------------------------------------------------- AngularSection --

def _legacy_get_section(beam, ang_min, ang_max):
    """The pre-refactor ``utils.get_section`` logic, kept as a regression reference."""
    ang_min %= 2 * np.pi
    ang_max %= 2 * np.pi
    theta = np.mod(np.arctan2(beam.y, beam.x), 2 * np.pi)
    if ang_min < ang_max:
        sec = (theta >= ang_min) & (theta < ang_max)
    else:
        sec = (theta >= ang_min) | (theta < ang_max)
    return beam.field * sec[None, :, :]


ANGLE_CASES = [
    (0, np.pi / 2), (np.pi / 2, np.pi), (3 * np.pi / 2, np.pi / 2), (-np.pi / 4, np.pi / 4),
    (0, 2 * np.pi), (np.pi, np.pi), (7 * np.pi / 3, 8 * np.pi / 3), (np.pi / 6, 11 * np.pi / 6),
]


@pytest.mark.parametrize("ang_min,ang_max", ANGLE_CASES)
def test_angular_section_matches_legacy_get_section(random_beam_factory, ang_min, ang_max):
    b = random_beam_factory()
    assert np.array_equal((b * AngularSection(ang_min, ang_max)).field,
                          _legacy_get_section(b, ang_min, ang_max))


def test_quadrants_partition_the_plane(beam_factory):
    b = beam_factory(Dx=65)  # odd grid: a pixel sits exactly on the origin and on the axes
    edges = [k * np.pi / 2 for k in range(5)]
    count = sum(AngularSection(edges[k], edges[k + 1]).array(b).astype(int) for k in range(4))
    assert np.all(count == 1)


def test_section_crossing_the_x_axis(beam_factory):
    b = beam_factory(Dx=65)
    right = AngularSection(3 * np.pi / 2, np.pi / 2).array(b)
    left = AngularSection(np.pi / 2, 3 * np.pi / 2).array(b)
    assert np.array_equal(right, ~left)
    x_full = np.broadcast_to(b.x, right.shape)
    assert np.all(x_full[right] >= 0) and np.all(x_full[left] <= 0)


def test_angles_are_wrapped(beam_factory):
    b = beam_factory(Dx=64)
    reference = AngularSection(2 * np.pi - 0.3, 1.1).array(b)
    assert np.array_equal(AngularSection(-0.3, 1.1).array(b), reference)
    assert np.array_equal(AngularSection(-0.3 + 4 * np.pi, 1.1 + 2 * np.pi).array(b), reference)


def test_equal_wrapped_angles_cover_the_whole_plane(beam_factory):
    b = beam_factory(Dx=32)
    assert AngularSection(0, 2 * np.pi).array(b).all()


def test_section_center(beam_factory):
    b = beam_factory(Dx=64)
    c = (1e-3, -2e-3)
    arr = AngularSection(0, np.pi / 2, center=c).array(b)
    x, y = np.broadcast_arrays(b.x - c[0], b.y - c[1])
    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
    assert np.array_equal(arr, (theta >= 0) & (theta < np.pi / 2))


def test_angular_section_preserves_all_polarization_components(random_beam_factory):
    b = random_beam_factory(pol_dim=3)
    mask = AngularSection(0.2, 1.4)
    inside = mask.array(b)
    out = b * mask
    assert np.array_equal(out.field[:, inside], b.field[:, inside])
    assert not out.field[:, ~inside].any()


# ---------------------------------------------------------- restrict_mask --

def test_restrict_spatial_mask(random_beam_factory):
    b = random_beam_factory(); orig = b.field.copy()
    region = AngularSection(0, np.pi)
    inside = region.array(b)
    out = b * restrict_mask(Lens(0.5), region)
    assert np.allclose(out.field[:, inside], (b * Lens(0.5)).field[:, inside])
    assert np.array_equal(out.field[:, ~inside], orig[:, ~inside])  # untouched, not blocked
    assert np.array_equal(b.field, orig)


def test_restrict_polarization_mask(random_beam_factory):
    b = random_beam_factory(pol_dim=2); orig = b.field.copy()
    region, qwp = AngularSection(0, np.pi), QWP(np.pi / 4)
    inside = region.array(b)
    reference = b * qwp
    assert not np.allclose(reference.field[:, inside], orig[:, inside])  # the plate does something
    out = b * restrict_mask(qwp, region)
    assert np.allclose(out.field[:, inside], reference.field[:, inside])
    assert np.array_equal(out.field[:, ~inside], orig[:, ~inside])


def test_restrict_polarizer_leaves_ez_untouched(random_beam_factory):
    b = random_beam_factory(pol_dim=3); orig = b.field.copy()
    region = Iris(radius=2e-3)
    inside = region.array(b)
    out = b * restrict_mask(Polarizer(0.3, "D"), region)
    assert np.array_equal(out.field[2], orig[2])
    assert np.array_equal(out.field[:, ~inside], orig[:, ~inside])
    assert np.allclose(out.field[:, inside], (b * Polarizer(0.3, "D")).field[:, inside])


def test_restrict_jones_mask_on_scalar_beam_is_noop(beam_factory):
    b = beam_factory(pol_dim=1, Dx=32); b.field[:] = 1 + 2j
    out = b * restrict_mask(QWP(0.2), AngularSection(0, np.pi))
    assert np.array_equal(out.field, b.field) and out is not b


def test_restrict_accepts_either_argument_order(random_beam_factory):
    b = random_beam_factory()
    region, lens = AngularSection(0, np.pi), Lens(0.5)
    assert np.array_equal((b * restrict_mask(lens, region)).field,
                          (b * restrict_mask(region, lens)).field)


def test_restrict_two_boolean_masks_uses_second_as_region(random_beam_factory):
    b = random_beam_factory()
    aperture, region = Iris(radius=2e-3), HSlit(size=1e-3)
    inside = region.array(b)
    out = b * restrict_mask(aperture, region)
    assert np.array_equal(out.field[:, inside], (b * aperture).field[:, inside])
    assert np.array_equal(out.field[:, ~inside], b.field[:, ~inside])


def test_restrict_with_complementary_region_and_nesting(random_beam_factory):
    b = random_beam_factory(pol_dim=2); orig = b.field.copy()
    qwp = QWP(np.pi / 4)
    comp = b * restrict_mask(qwp, AngularSection(0, np.pi, complementary=True))
    lower = ~AngularSection(0, np.pi).array(b)
    assert np.array_equal(comp.field[:, ~lower], orig[:, ~lower])
    assert np.allclose(comp.field[:, lower], (b * qwp).field[:, lower])

    both = AngularSection(0, np.pi).array(b) & Iris(radius=2e-3).array(b)
    nested = b * restrict_mask(restrict_mask(qwp, AngularSection(0, np.pi)), Iris(radius=2e-3))
    assert np.allclose(nested.field[:, both], (b * qwp).field[:, both])
    assert np.array_equal(nested.field[:, ~both], orig[:, ~both])


def test_restrict_returns_a_region_mask_and_validates_arguments():
    assert isinstance(restrict_mask(Lens(1.0), Iris(radius=1e-3)), RegionMask)
    with pytest.raises(TypeError):
        restrict_mask(Lens(1.0), QWP(0))          # neither is a BooleanMask
    with pytest.raises(TypeError):
        restrict_mask("not a mask", Iris(radius=1e-3))
