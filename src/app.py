import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout

def make_flat_field_noise_u16(h: int, w: int, mean: int = 30000, sigma: int = 2500) -> np.ndarray:
    """
    Create a flat-field-like X-ray noise image as uint 16.
    mean/sigma are in 16-bit intensity units.

    Args:
        h (int): _description_
        w (int, optional): _description_. Defaults to 30000.
        sigma (int, optional): _description_. Defaults to 2500.

    Returns:
        np.ndarray: _description_
    """
    img = np.random.normal(loc=mean, scale=sigma, size=(h, w)).astype(np.float32)
    img = np.clip(img, 0, 65535).astype(np.uint16)
    return img

def u16_to_qimage_grayscale8(img_u16: np.ndarray) -> QImage:
    """
    Convert uint16 (H,W) to 8-bit grayscale QImage for display.
    We normalize by min/max for now (later: window/level).

    Args:
        img_u16 (np.ndarray): _description_

    Returns:
        QImage: _description_
    """
    if img_u16.dtype != np.uint16 or img_u16.ndim != 2:
        raise ValueError("Expected (H,W) uint16 image")
    
    vmin = int(img_u16.min())
    vmax = int(img_u16.max())
    if vmax == vmin:
        vmax = vmin + 1

    img8 = ((img_u16.astype(np.float32) - vmin) * (255.0 / (vmax - vmin))).astype(np.uint8)
              
    h, w = img8.shape
    # QImage needs bytes. Keep a reference by copying via .copy() at end.
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()
    return qimg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fluoro X-ray Sim GUI (Step 1)")
        self.resize(1000, 700)

        root = QWidget()
        layout = QVBoxLayout(root)

        self.image_label = QLabel("Image Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #111; color: #bbb; border: 1px solid #333;") 

        layout.addWidget(self.image_label, stretch=1)
        self.setCentralWidget(root)

        # Generate and show the first flat-field noise image
        img_u16 = make_flat_field_noise_u16(768, 768, mean=32000, sigma=2200)
        qimg = u16_to_qimage_grayscale8(img_u16)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # Rescale pixmap to keep aspect ratio on window resize
        pm = self.image_label.pixmap()
        if pm is not None and not pm.isNull():
            self.image_label.setPixmap(pm.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()