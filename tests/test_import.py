"""The package must import cleanly on every platform, including headless Linux."""
import importlib
import subprocess
import sys

import pytest

PUBLIC_NAMES = [
    # beam + modes
    "Beam", "hg", "hg_astigmatic", "lg", "nbessel", "gbessel", "lg_prod",
    "frac_oam", "frac_oam_qs", "IG_even", "IG_odd", "HInceG", "circle", "square",
    "triangle", "lp", "lp_hel",
    # masks
    "Lens", "AstigmaticLens", "TiltedLens", "ZernikeMask", "Iris", "SquareAperture",
    "TriangleAperture", "HSlit", "VSlit", "DoubleSlit", "CrossSlit",
    "AngularSection", "RegionMask", "restrict_mask",
    # polarization
    "HWP", "QWP", "Polarizer", "J_rot", "J_hwp", "J_qwp",
    # tools
    "overlap", "int_overlap", "hg_proj", "lg_proj", "hg_basis", "lg_basis", "bessel_basis",
]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_is_exported(name):
    import structured_optics

    assert hasattr(structured_optics, name)


@pytest.mark.parametrize(
    "module",
    ["utils", "modes", "masks", "polarization", "tools", "diagnostics",
     "prop_methods", "bluestein", "hologram", "struct_opt"],
)
def test_submodule_imports(module):
    importlib.import_module(f"structured_optics.{module}")


def test_import_does_not_need_gui_libraries(tmp_path):
    """`import structured_optics` must work without tkinter/screeninfo (headless servers, CI)."""
    code = (
        "import sys\n"
        "sys.modules['tkinter'] = None\n"
        "sys.modules['screeninfo'] = None\n"
        "import structured_optics\n"
        "structured_optics.Beam(nix=1e-3, Dx=8)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
