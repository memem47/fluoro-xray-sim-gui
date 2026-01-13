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
    QComboBox,
    QSplitter,
    QScrollArea,
    QGroupBox,
    QFormLayout,
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

def lowfreq_approx_u16(img_u16: np.ndarray, downsample: int = 16) -> np.ndarray:
    """
    Create a low-frequency approximation by downsampling then upsampling.
    No extra dependencies, cheap enough for real-time.

    Args:
        img_u16 (np.ndarray): _description_
        downsample (int, optional): _description_. Defaults to 16.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if downsample <= 1:
        return img_u16
    
    h, w = img_u16.shape
    ds = downsample

    # crop to multiple of ds for simplicity
    h2 = (h // ds) * ds
    w2 = (w // ds) * ds
    img = img_u16[:h2, :w2].astype(np.float32)

    small = img.reshape(h2 // ds, ds, w2 // ds, ds).mean(axis=(1,3))
    up = np.repeat(np.repeat(small, ds, axis=0), ds, axis=1)

    out = np.zeros((h, w), dtype=np.float32)
    out[:h2, :w2] = up
    if h2 < h:
        out[h2:, :w2] = up[-1:, :]
    if w2 < w:
        out[:h2, w2:] = up[:, -1:]
    if h2 < h and w2 < w:
        out[h2:, w2:] = up[-1:, -1:]

    return np.clip(out, 0, 65535).astype(np.uint16)

def apply_scatter_veil_u16(base_u16: np.ndarray, strength: float, downsample: int = 16) -> np.ndarray:
    """
    Add a low-frequency `veil` component (scatter-like) to the base signal.
    strength: 0..1

    Args:
        img_u16 (np.ndarray): _description_
        strength (float): _description_
        downsample (int, optional): _description_. Defaults to 16.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    lf = lowfreq_approx_u16(base_u16, downsample=downsample).astype(np.float32)
    base = base_u16.astype(np.float32)

    # Additive veil: base + strength * lowfreq
    out = base + strength * lf
    return np.clip(out, 0, 65535).astype(np.uint16)

def apply_motion_blur_u16(
        current_u16: np.ndarray,
        prev_u16: np.ndarray | None,
        alpha: float,
) -> np.ndarray:
    """
    Simple exponential motion blur using previous frame.
    out = (1 - alpha) * current + alpha * prev

    Args:
        current_u16 (np.ndarray): _description_
        prev_u16 (np.ndarray | None): _description_
        alpha (float): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    if prev_u16 is None:
        return current_u16
    
    alpha = float(np.clip(alpha, 0.0, 0.99))
    cur = current_u16.astype(np.float32)
    prev = prev_u16.astype(np.float32)

    out = (1.0 - alpha) * cur + alpha * prev
    return np.clip(out, 0, 65535).astype(np.uint16)

def smoothstep01(x: np.ndarray) -> np.ndarray:
    """
    Smoothstep function from 0 to 1.
    x: assumed in [0,1]
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def make_collimation_mask(h: int, w: int, shape: str, size: float, soft_px: int) -> np.ndarray:
    """
    Returns float32 mask in [0,1]. 1=inside field, 0=outside.
    shape: "Rectangle" or "Circle"
    size: 0.1..1.0 (relative aperture size)
    soft_px: soft edge width in pixels (0 => hard edge)

    Args:
        h (int): _description_
        w (int): _description_
        shape (str): _description_
        size (float): _description_
        soft_px (int): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    size = float(np.clip(size, 0.1, 1.0))
    soft = int(max(soft_px, 0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x = xx - cx
    y = yy - cy

    if shape == "Circle":
        r = min(h, w) * 0.5 * size
        dist = np.sqrt(x**2 + y**2)
    else:
        # Rectangle (axis-aligned)                 
        hx = w * 0.5 * size
        hy = h * 0.5 * size
        # signed distance to rectangle boundary (positive outside)
        dx = np.abs(x) - hx
        dy = np.abs(y) - hy
        dist = np.maximum(dx, dy)

    if soft <= 0:
        return (dist <= 0).astype(np.float32)

    # Soft edge: map dist in [-soft, soft] to [1..0]
    t = (-dist + soft) / (2.0 * soft)
    return smoothstep01(t).astype(np.float32)

def apply_collimation_u16(img_u16: np.ndarray, mask01: np.ndarray, 
                  outside_level: float) -> np.ndarray:
    """
    Apply collimation  as multiplicative attenuation with a floor outside_level.
    out = img * (outside + (1-outside)*mask)

    Args:
        img_u16 (np.ndarray): _description_
        mask01 (np.ndarray): _description_
        outside (float): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    outside = float(np.clip(outside_level, 0.0, 1.0))
    img = img_u16.astype(np.float32)
    m = mask01.astype(np.float32)
    gain = outside + (1.0 - outside) * m
    out = img * gain
    return np.clip(out, 0, 65535).astype(np.uint16)

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

        # --- Scatter parameters (Step 4-2) ---
        self.enable_scatter = True
        self.scatter_strength = 0.25
        self.scatter_downsample = 16
        
        # --- Motion blur parameters (Step 4-3) ---
        self.enable_motion_blur = True
        self.motion_blur_alpha = 0.5
        self.prev_frame_u16 = None

        # --- Collimation parameters (Step 4-4) ---
        self.enable_collimation = True
        self.collimation_shape = "Rectangle"  # "Rectangle" or "Circle"
        self.collimation_size = 0.85
        self.collimation_soft_px = 24
        self.collimation_outside = 0.02

        self._collimation_mask = None
        self._collimation_mask_key = None

        # --- UI ---
        root = QWidget()

        root_layout = QVBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        # Left image view
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)

        # Image area
        self.image_label = QLabel("Image Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #111; color: #bbb; border: 1px solid #333;") 

        left_layout.addWidget(self.image_label, stretch=1)
    
        splitter.addWidget(left_widget)

        # Right: control panel (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8,8,8,8)
        right_layout.setSpacing(12)

        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)

        # Splitter size policy (optional but recommended)
        splitter.setStretchFactor(0, 3)  # left grows more
        splitter.setStretchFactor(1, 1)  # right grows less

        splitter.setSizes([800, 200])

        root_layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(root)

        gb_display = QGroupBox("Display")
        display_form = QFormLayout(gb_display)

        # WL row
        wl_row = QHBoxLayout()
        self.wl_label = QLabel()
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(0, 65535)
        self.wl_slider.setValue(self.wl)
        self.wl_slider.valueChanged.connect(self.on_wl_changed)

        wl_line = QWidget()
        wl_line_l = QHBoxLayout(wl_line)
        wl_line_l.setContentsMargins(0,0,0,0)
        wl_line_l.addWidget(self.wl_slider, stretch=1)
        wl_line_l.addWidget(self.wl_label)
        display_form.addRow("WL", wl_line)

        # WW row
        ww_row = QHBoxLayout()
        self.ww_label = QLabel()
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(0, 65535)
        self.ww_slider.setValue(self.ww)
        self.ww_slider.valueChanged.connect(self.on_ww_changed)

        ww_line = QWidget()
        ww_line_l = QHBoxLayout(ww_line)
        ww_line_l.setContentsMargins(0,0,0,0)
        ww_line_l.addWidget(self.ww_slider, stretch=1)
        ww_line_l.addWidget(self.ww_label)
        display_form.addRow("WW", ww_line)

        right_layout.addWidget(gb_display)

        # Record now
        gb_record = QGroupBox("Recording")
        record_layout = QVBoxLayout(gb_record)

        self.btn_record_5s = QPushButton("Record 5s (MP4)")
        self.btn_record_5s.clicked.connect(self.on_record_5s)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)
            
        record_layout.addWidget(self.btn_record_5s)
        record_layout.addWidget(self.status_label)

        right_layout.addWidget(gb_record)


        # Poisson (quantum) noise controls
        gb_poisson = QGroupBox("Quantum Noise (Poisson)")
        poisson_layout = QVBoxLayout(gb_poisson)

        self.poisson_checkbox = QCheckBox("Quantum noise (Poisson)")
        self.poisson_checkbox.setChecked(self.enable_poisson)
        self.poisson_checkbox.toggled.connect(self.on_poisson_toggled)
    
        self.poisson_label = QLabel()
        self.poisson_slider = QSlider(Qt.Horizontal)
        self.poisson_slider.setRange(1000, 65000)   # I0 range
        self.poisson_slider.setValue(self.poisson_i0)
        self.poisson_slider.valueChanged.connect(self.on_poisson_i0_changed)
        
        poisson_layout.addWidget(self.poisson_checkbox)
        poisson_layout.addWidget(self.poisson_label)
        poisson_layout.addWidget(self.poisson_slider)

        right_layout.addWidget(gb_poisson)

        # Scatter controls
        gb_scatter = QGroupBox("Scatter Veil")
        gb_scatter_layout = QVBoxLayout(gb_scatter)

        self.scatter_checkbox = QCheckBox("Scatter veil")
        self.scatter_checkbox.setChecked(self.enable_scatter)
        self.scatter_checkbox.toggled.connect(self.on_scatter_toggled)

        self.scatter_label = QLabel()
        self.scatter_slider = QSlider(Qt.Horizontal)
        self.scatter_slider.setRange(0, 100)   # Scatter strength in %
        self.scatter_slider.setValue(int(self.scatter_strength * 100))
        self.scatter_slider.valueChanged.connect(self.on_scatter_strength_changed)

        gb_scatter_layout.addWidget(self.scatter_checkbox)
        gb_scatter_layout.addWidget(self.scatter_label)
        gb_scatter_layout.addWidget(self.scatter_slider)

        # Motion blur controls
        gb_motion = QGroupBox("Motion Blur")
        gb_motion_layout = QVBoxLayout(gb_motion)

        self.motion_checkbox = QCheckBox("Motion blur")
        self.motion_checkbox.setChecked(self.enable_motion_blur)
        self.motion_checkbox.toggled.connect(self.on_motion_blur_toggled)

        self.motion_label = QLabel()
        self.motion_slider = QSlider(Qt.Horizontal)
        self.motion_slider.setRange(0, 90)
        self.motion_slider.setValue(int(self.motion_blur_alpha * 100))
        self.motion_slider.valueChanged.connect(self.on_motion_blur_alpha_changed)

        gb_motion_layout.addWidget(self.motion_checkbox)
        gb_motion_layout.addWidget(self.motion_label)
        gb_motion_layout.addWidget(self.motion_slider)

        right_layout.addWidget(gb_motion)

        # Collimation controls
        gb_col = QGroupBox("Collimation")
        gb_col_layout = QVBoxLayout(gb_col)

        self.col_checkbox = QCheckBox("Collimation")
        self.col_checkbox.setChecked(self.enable_collimation)
        self.col_checkbox.toggled.connect(self.on_collimation_toggled)

        self.col_shape = QComboBox()
        self.col_shape.addItems(["Rectangle", "Circle"])
        self.col_shape.setCurrentText(self.collimation_shape)
        self.col_shape.currentTextChanged.connect(self.on_collimation_shape_changed)

        self.col_size_label = QLabel()
        self.col_size = QSlider(Qt.Horizontal)
        self.col_size.setRange(10, 100)   # 0.1 .. 1.0
        self.col_size.setValue(int(self.collimation_size * 100))
        self.col_size.valueChanged.connect(self.on_collimation_size_changed)
        
        self.col_soft_label = QLabel()
        self.col_soft = QSlider(Qt.Horizontal)
        self.col_soft.setRange(0, 200)   # in pixels
        self.col_soft.setValue(int(self.collimation_soft_px))
        self.col_soft.valueChanged.connect(self.on_collimation_soft_changed)

        self.col_out_label = QLabel()
        self.col_out = QSlider(Qt.Horizontal)
        self.col_out.setRange(0, 30)   # 0 .. 1
        self.col_out.setValue(int(self.collimation_outside * 100))
        self.col_out.valueChanged.connect(self.on_collimation_out_changed)

        gb_col_layout.addWidget(self.col_checkbox)
        gb_col_layout.addWidget(QLabel("Shape"))
        gb_col_layout.addWidget(self.col_shape)
        gb_col_layout.addWidget(QLabel("Size"))
        gb_col_layout.addWidget(self.col_size)
        gb_col_layout.addWidget(self.col_size_label)
        gb_col_layout.addWidget(QLabel("Soft edge (px)"))
        gb_col_layout.addWidget(self.col_soft)
        gb_col_layout.addWidget(self.col_soft_label)
        gb_col_layout.addWidget(QLabel("Outside level"))
        gb_col_layout.addWidget(self.col_out)
        gb_col_layout.addWidget(self.col_out_label)
        right_layout.addWidget(gb_col)

        right_layout.addStretch(1)

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
        self.scatter_label.setText(f"{int(self.scatter_strength*100):3d}%")
        self.motion_label.setText(f"{int(self.motion_blur_alpha*100):3d}%")
        self.col_size_label.setText(f"{int(self.collimation_size * 100):3d}%")
        self.col_soft_label.setText(f"{int(self.collimation_soft_px):3d}")
        self.col_out_label.setText(f"{int(self.collimation_outside * 100):2d}%")

    def on_wl_changed(self, value: int):
        self.wl = int(value)
        self.sync_labels()
        self.render()
        
    def on_ww_changed(self, value: int):
        self.ww = int(value)
        self.sync_labels()
        self.render()

    def on_poisson_toggled(self, checked: bool):
        self.enable_poisson = checked

    def on_poisson_i0_changed(self, value: int):
        self.poisson_i0 = int(value)
        self.sync_labels()
    
    def on_scatter_toggled(self, checked: bool):
        self.enable_scatter = checked

    def on_scatter_strength_changed(self, value: int):
        self.scatter_strength = float(value) / 100.0
        self.sync_labels()
    
    def on_motion_blur_toggled(self, checked: bool):
        self.enable_motion_blur = checked
        if not self.enable_motion_blur:
            self.prev_frame_u16 = None

    def on_motion_blur_alpha_changed(self, value: int):
        self.motion_blur_alpha = float(value) / 100.0
        self.sync_labels()

    def invalidate_collimation_mask(self):
        self._collimation_mask = None
        self._collimation_mask_key = None

    def on_collimation_toggled(self, checked: bool):
        self.enable_collimation = checked
        # no need to invalidate; we can keep the mask cached

    def on_collimation_shape_changed(self, text: str):
        self.collimation_shape = text
        self.invalidate_collimation_mask()

    def on_collimation_size_changed(self, value: int):
        self.collimation_size = float(value) / 100.0
        self.sync_labels()
        self.invalidate_collimation_mask()

    def on_collimation_soft_changed(self, value: int):
        self.collimation_soft_px = int(value)
        self.sync_labels()
        self.invalidate_collimation_mask()

    def on_collimation_out_changed(self, value: int):
        self.collimation_outside = float(value) / 100.0
        self.sync_labels()
        # outside is applied after mask; no need to invalidate mask

    def render(self):
        qimg = u16_to_qimage_grayscale8_wlww(self.img_u16, self.wl, self.ww)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def get_collimation_mask(self) -> np.ndarray:
        h, w = self.img_u16.shape
        key = (h, w, self.collimation_shape, round(self.collimation_size, 4), int(self.collimation_soft_px))
        if self._collimation_mask is None or self._collimation_mask_key != key:
            self._collimation_mask = make_collimation_mask(
                h, w,
                self.collimation_shape,
                self.collimation_size,
                self.collimation_soft_px
            )
            self._collimation_mask_key = key
        return self._collimation_mask

    def on_tick(self):
        # Base "flat-field" signal (later: physics pipeline will replace this)
        base = make_flat_field_noise_u16(768, 768, mean=32000, sigma=800)  # sigmaは少し小さめでもOK

        if self.enable_scatter:
            base = apply_scatter_veil_u16(
                base, 
                self.scatter_strength, 
                self.scatter_downsample
            )

        if self.enable_collimation:
            mask = self.get_collimation_mask()
            base = apply_collimation_u16(
                base,
                mask,
                self.collimation_outside
            )
        
        if self.enable_motion_blur:
            base = apply_motion_blur_u16(
                base,
                self.prev_frame_u16,
                self.motion_blur_alpha
            )
            self.prev_frame_u16 = base.copy()
        else:
            self.prev_frame_u16 = None
            
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