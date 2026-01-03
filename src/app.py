import sys
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, 
    QLabel, 
    QMainWindow, 
    QWidget, 
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QCheckBox,
)

import imageio.v2 as imageio

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

def u16_to_u8_wlww(img_u16: np.ndarray, wl:int, ww: int) -> np.ndarray:
    """
    Convert uint16 (H,W) to uint8 (H,W) using Window/Level.
    This returns a numpy array for recording/export.
    """
    if img_u16.dtype != np.uint16 or img_u16.ndim != 2:
        raise ValueError("Expected (H,W) uint16 image")
    
    ww = max(int(ww), 1)
    wl = int(wl)

    lower = wl - ww / 2.0
    img_f = img_u16.astype(np.float32)

    img8 = (img_f - lower) * (255.0 / ww)
    img8 = np.clip(img8, 0, 255).astype(np.uint8)
    return img8

def apply_poisson_noise_u16(base_u16: np.ndarray, i0: int) -> np.ndarray:
    """
    Apply Poisson noise assuming the signal represents expected photon counts.
    i0 controls the scale of counts. Larger i0 -> relatively lower noise.
    """
    i0 = max(int(i0), 1)

    # Normalize base image to [0,1] then scale to counts ~ i0
    base = base_u16.astype(np.float32) / 65535.0
    lam = np.clip(base * i0, 0, None)

    noisy = np.random.poisson(lam).astype(np.float32)

    # Rescale back to uint16 range
    out = np.clip(noisy / i0 * 65535.0, 0, 65535).astype(np.uint16)
    return out

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

        # --- Recording state ---
        self.is_recording = False
        self.record_frames: list[np.ndarray] = []
        self.record_target_frames = 0
        self.record_path = ""

        # --- Physics parameters (Step 4-1) ---
        self.enable_poisson = True
        self.poisson_i0 = 30000 # mean intensity for Poisson noise strength

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

        # Record now
        record_row = QHBoxLayout()
        self.btn_record_5s = QPushButton("Record 5s (MP4)")
        self.btn_record_5s.clicked.connect(self.on_record_5s)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)

        record_row.addWidget(self.btn_record_5s)
        record_row.addWidget(self.status_label, stretch=1)
        controls_layout.addLayout(record_row)

        # Poisson (quantum) noise controls
        poisson_row = QHBoxLayout()

        self.poisson_checkbox = QCheckBox("Quantum noise (Poisson)")
        self.poisson_checkbox.setChecked(self.enable_poisson)
        self.poisson_checkbox.stateChanged.connect(self.on_poisson_toggled)

        self.poisson_label = QLabel()
        self.poisson_slider = QSlider(Qt.Horizontal)
        self.poisson_slider.setRange(1000, 65000)   # I0 range
        self.poisson_slider.setValue(self.poisson_i0)
        self.poisson_slider.valueChanged.connect(self.on_poisson_i0_changed)

        poisson_row.addWidget(self.poisson_checkbox)
        poisson_row.addWidget(QLabel("I0"))
        poisson_row.addWidget(self.poisson_slider, stretch=1)
        poisson_row.addWidget(self.poisson_label)

        controls_layout.addLayout(poisson_row)

        # Initial label text + render
        self.sync_labels()
        self.render()

        # --- Frame loop (30 FPS) ---
        self.fps = 30
        interval_ms = int(1000 / self.fps)

        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(interval_ms)

    def sync_labels(self):
        self.wl_label.setText(f"{self.wl:5d}")
        self.ww_label.setText(f"{self.ww:5d}")
        self.poisson_label.setText(f"{self.poisson_i0:5d}")

    def on_wl_changed(self, value: int):
        self.wl = int(value)
        self.sync_labels()
        self.render()
        
    def on_ww_changed(self, value: int):
        self.ww = int(value)
        self.sync_labels()
        self.render()

    def on_poisson_toggled(self, state: int):
        self.enable_poisson = (state == Qt.Checked)

    def on_poisson_i0_changed(self, value: int):
        self.poisson_i0 = int(value)
        self.sync_labels()

    def render(self):
        qimg = u16_to_qimage_grayscale8_wlww(self.img_u16, self.wl, self.ww)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def on_tick(self):
        # Base "flat-field" signal (later: physics pipeline will replace this)
        base = make_flat_field_noise_u16(768, 768, mean=32000, sigma=800)  # sigmaは少し小さめでもOK

        # Apply Poisson noise (quantum noise)
        if self.enable_poisson:
            self.img_u16 = apply_poisson_noise_u16(base, self.poisson_i0)
        else:
            self.img_u16 = base

        # If recording, store current display frame (uint8)
        if self.is_recording:
            frame_u8 = u16_to_u8_wlww(self.img_u16, self.wl, self.ww)
            self.record_frames.append(frame_u8)

            remaining = self.record_target_frames - len(self.record_frames)
            self.status_label.setText(f"Recording... {remaining} frames left")

            if len(self.record_frames) >= self.record_target_frames:
                self.finish_recording()

        # Update display
        self.render()


    def on_record_5s(self):
        if self.is_recording:
            return # Safety: ignore while recoding
        
        # Choose output path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save MP4",
            "fluoro_recoring.mp4",
            "MP4 Video (*.mp4)"
        )
        if not path:
            return
        
        # Initialize recoding
        seconds = 5
        self.record_path = path
        self.record_frames = []
        self.record_target_frames = int(self.fps * seconds)
        self.is_recording = True

        self.btn_record_5s.setEnabled(False)
        self.status_label.setText(f"Recording... {self.record_target_frames} frames left")

    def finish_recording(self):
        # Stop recoding state first (so UI doesn't keep appending)
        self.is_recording = False
        self.btn_record_5s.setEnabled(True)

        try:
            self.save_mp4(self.record_path, self.record_frames, fps=self.fps)
            self.status_label.setText("Saved MP4")
            QMessageBox.information(self, "Export", f"Saved MP4:\n{self.record_path}")
        except Exception as e:
            self.status_label.setText("Export failed")
            QMessageBox.critical(self, "Export failed", str(e))
        finally:
            self.record_frames = []
            self.record_target_frames = 0
            self.record_path = ""

    def save_mp4(self, path: str, frames_u8: list[np.ndarray], fps: int):
        """
        Save list of uint8 grayscale frames (H,W) to MP4.
        Many MP4 encoders expect RGB, so we replicate grayscale to 3 channels.

        Args:
            path (str): _description_
            frames_u8 (list[np.ndarray]): _description_
            fps (int): _description_
        """
        if not frames_u8:
            raise ValueError("No frames to save")
        
        # Convert to RGB frames for broad compatibility
        rgb_frames = []
        for f in frames_u8:
            if f.dtype != np.uint8 or f.ndim != 2:
                raise ValueError("Expected uint8 grayscale frames (H,W).")
            rgb = np.repeat(f[:,:,None], 3, axis=2)  # (H,W,3)
            rgb_frames.append(rgb)
        
        writer = imageio.get_writer(path, fps=fps)
        try:
            for rgb in rgb_frames:
                writer.append_data(rgb)
        finally:
            writer.close()

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