"""import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class My_SLM(QLabel):
    def __init__(self, screen_index=0):
        self.app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()

        screens = self.app.screens()
        if screen_index >= len(screens):
            raise ValueError("Invalid screen index")

        screen = screens[screen_index]

        # Force native window creation (CRITICAL on Linux)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
        )

        self.show()  # must come before windowHandle()

        self.windowHandle().setScreen(screen)
        geo = screen.geometry()

        # Store logical screen size
        self.width = geo.width()
        self.height = geo.height()
        self.setGeometry(geo)

        self.showFullScreen()

        self._img_ref = None


    def update_image(self, img: np.ndarray):
        
        #img:
        #  - (H, W) uint8 → grayscale
        #  - (H, W, 3) uint8 → RGB
        
        if img.dtype != np.uint8:
            raise TypeError("Image must be uint8")

        self._img_ref = img  # keep reference alive

        if img.ndim == 2:
            h, w = img.shape
            stride = img.strides[0]

            qimg = QImage(
                img.data,
                w,
                h,
                stride,
                QImage.Format.Format_Grayscale8
            )

        elif img.ndim == 3 and img.shape[2] == 3:
            h, w, _ = img.shape
            stride = img.strides[0]

            qimg = QImage(
                img.data,
                w,
                h,
                stride,
                QImage.Format.Format_RGB888
            )
        else:
            raise ValueError("Image must be (H,W) or (H,W,3)")

        self.setPixmap(QPixmap.fromImage(qimg))
        self.app.processEvents()

    def close(self):
        if not self.isVisible():
            return
        self.hide()
        self.setParent(None)   # detach from Qt ownership
        self.deleteLater() """

import tkinter as tk
import numpy as np
from screeninfo import get_monitors


class My_SLM:
    """
    Fullscreen numpy-array display on a chosen physical monitor.
    Uses plain tkinter (stdlib) - the image is shown as ordinary
    CPU-composited window content, the same way PyQt6's QLabel/QPixmap
    did. Unlike an OpenGL surface (pyglet/GLFW), this never becomes a
    GPU flip-model surface, so it doesn't trigger the driver-level
    scanout switch that causes the display to briefly flash/resync.
    """

    def __init__(self, screen_index: int = 0):
        monitors = get_monitors()
        if screen_index >= len(monitors):
            raise ValueError("Invalid screen index")

        m = monitors[screen_index]
        self.width, self.height = m.width, m.height

        self.root = tk.Tk()
        self.root.overrideredirect(True)  # borderless
        self.root.geometry(f"{self.width}x{self.height}+{m.x}+{m.y}")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")

        self.label = tk.Label(self.root, bd=0, highlightthickness=0, bg="black")
        self.label.pack(fill="both", expand=True)

        self.root.update_idletasks()
        self.root.update()

        self._photo_ref = None  # keep the PhotoImage alive (Tk drops it otherwise)
        self._img_ref = None

    def update_image(self, img: np.ndarray):
        """
        img:
          - (H, W) uint8      -> grayscale
          - (H, W, 3) uint8   -> RGB
        """
        if img.dtype != np.uint8:
            raise TypeError("Image must be uint8")

        img = np.ascontiguousarray(img)
        self._img_ref = img  # keep a reference alive

        if img.ndim == 2:
            h, w = img.shape
            header = f"P5\n{w} {h}\n255\n".encode()
        elif img.ndim == 3 and img.shape[2] == 3:
            h, w, _ = img.shape
            header = f"P6\n{w} {h}\n255\n".encode()
        else:
            raise ValueError("Image must be (H,W) or (H,W,3)")

        # Build a raw PGM/PPM blob in memory - Tk's built-in photo image
        # reader parses this directly in C, so no PIL/Pillow dependency
        # and no per-pixel Python loop.
        data = header + img.tobytes()
        photo = tk.PhotoImage(data=data)

        self._photo_ref = photo
        self.label.configure(image=photo)

        self.root.update()

    def close(self):
        self.root.destroy()


