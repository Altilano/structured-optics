"""My_SLM display wrapper, tested against fake tkinter/screeninfo objects (no screen needed)."""
from types import SimpleNamespace

import numpy as np
import pytest

# Needs tkinter + screeninfo to be importable; skipped on Pythons built without Tk.
slmdisplay = pytest.importorskip("structured_optics.slmdisplay")

MONITORS = [
    SimpleNamespace(x=0, y=0, width=800, height=600),
    SimpleNamespace(x=800, y=0, width=1024, height=768),
]


class FakeRoot:
    def __init__(self):
        self.calls, self.destroyed = [], False

    def overrideredirect(self, flag): self.calls.append(("overrideredirect", flag))
    def geometry(self, spec): self.geometry_spec = spec
    def attributes(self, *args): self.calls.append(("attributes", args))
    def configure(self, **kw): self.calls.append(("configure", kw))
    def update_idletasks(self): pass
    def update(self): self.updates = getattr(self, "updates", 0) + 1
    def destroy(self): self.destroyed = True


class FakeLabel:
    def __init__(self, root, **kw): self.image = None
    def pack(self, **kw): pass
    def configure(self, **kw): self.image = kw.get("image", self.image)


class FakePhotoImage:
    def __init__(self, data=None): self.data = data


@pytest.fixture
def slm(monkeypatch):
    monkeypatch.setattr(slmdisplay, "get_monitors", lambda: MONITORS)
    monkeypatch.setattr(slmdisplay, "tk",
                        SimpleNamespace(Tk=FakeRoot, Label=FakeLabel, PhotoImage=FakePhotoImage))
    return slmdisplay.My_SLM(screen_index=1)


def test_uses_the_selected_monitor(slm):
    assert (slm.width, slm.height) == (1024, 768)
    assert slm.root.geometry_spec == "1024x768+800+0"


def test_invalid_screen_index(monkeypatch):
    monkeypatch.setattr(slmdisplay, "get_monitors", lambda: MONITORS)
    with pytest.raises(ValueError, match="Invalid screen index"):
        slmdisplay.My_SLM(screen_index=5)


def test_grayscale_image_is_sent_as_pgm(slm):
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    slm.update_image(img)
    data = slm.label.image.data
    assert data.startswith(b"P5\n4 3\n255\n") and data.endswith(img.tobytes())


def test_rgb_image_is_sent_as_ppm(slm):
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    slm.update_image(img)
    assert slm.label.image.data.startswith(b"P6\n3 2\n255\n")


def test_non_contiguous_images_are_accepted(slm):
    img = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    slm.update_image(img)
    assert slm.label.image.data.endswith(np.ascontiguousarray(img).tobytes())


def test_image_validation(slm):
    with pytest.raises(TypeError):
        slm.update_image(np.zeros((4, 4), dtype=float))
    with pytest.raises(ValueError):
        slm.update_image(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        slm.update_image(np.zeros(4, dtype=np.uint8))


def test_close_destroys_the_window(slm):
    slm.close()
    assert slm.root.destroyed
