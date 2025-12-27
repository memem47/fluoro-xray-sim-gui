import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, 
    QLabel, 
    QMainWindow, 
    QWidget, 
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
)

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

def u16_to_qimage_grayscale8_wlww(img_u16: np.ndarray, wl:int, ww: int) -> QImage:
    """
    Convert uint16 (H,W) to 8-bit grayscale QImage for display using Window/Level.
    lower = WL - WW/2, uppder = WL + WW/2

    Args:
        img_u16 (np.ndarray): _description_

    Returns:
        QImage: _description_
    """
    if img_u16.dtype != np.uint16 or img_u16.ndim != 2:
        raise ValueError("Expected (H,W) uint16 image")
    
    ww = max(int(ww), 1)
    wl = int(wl)

    lower = wl - ww / 2.0
    upper = wl + ww / 2.0

    # Map to [0, 255]
    img_f = img_u16.astype(np.float32)
    img8 = (img_f - lower) * (255.0 / ww)
    img8 = np.clip(img8, 0, 255).astype(np.uint8)
              
    h, w = img8.shape
    # QImage needs bytes. Keep a reference by copying via .copy() at end.
    qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()
    return qimg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fluoro X-ray Sim GUI (WL/WW)")
        self.resize(1100, 800)

        # --- Model (keep original 16 -bit image) ---
        self.img_u16 = make_flat_field_noise_u16(768, 768, mean=32000, sigma=2200)

        # Default WL/WW (reasonable defaults for 16-bit)
        self.wl = 32768
        self.ww = 65535

        # --- UI ---
        root = QWidget()
        main_layout = QVBoxLayout(root)

        # Image area
        self.image_label = QLabel("Image Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #111; color: #bbb; border: 1px solid #333;") 
        main_layout.addWidget(self.image_label, stretch=1)

        # Controls area
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        # WL row
        wl_row = QHBoxLayout()
        self.wl_label = QLabel()
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(0, 65535)
        self.wl_slider.setValue(self.wl)
        self.wl_slider.valueChanged.connect(self.on_wl_changed)
        wl_row.addWidget(QLabel("WL"))
        wl_row.addWidget(self.wl_slider, stretch=1)
        wl_row.addWidget(self.wl_label)
        controls_layout.addLayout(wl_row)

        # WW row
        ww_row = QHBoxLayout()
        self.ww_label = QLabel()
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(0, 65535)
        self.ww_slider.setValue(self.ww)
        self.ww_slider.valueChanged.connect(self.on_ww_changed)
        ww_row.addWidget(QLabel("WW"))
        ww_row.addWidget(self.ww_slider, stretch=1)
        ww_row.addWidget(self.ww_label)
        controls_layout.addLayout(ww_row)

        main_layout.addWidget(controls, stretch=0)        
        self.setCentralWidget(root)

        # Initial label text + render
        self.sync_labels()
        self.render()

    def sync_labels(self):
        self.wl_label.setText(f"{self.wl:5d}")
        self.ww_label.setText(f"{self.ww:5d}")

    def on_wl_changed(self, value: int):
        self.wl = int(value)
        self.sync_labels()
        self.render()
        
    def on_ww_changed(self, value: int):
        self.ww = int(value)
        self.sync_labels()
        self.render()

    def render(self):
        qimg = u16_to_qimage_grayscale8_wlww(self.img_u16, self.wl, self.ww)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        # Re-render on resize to keep scaling correct
        self.render()
        super().resizeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()