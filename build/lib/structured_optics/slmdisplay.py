import sys
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
        """
        img:
          - (H, W) uint8 → grayscale
          - (H, W, 3) uint8 → RGB
        """
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
        self.deleteLater() 