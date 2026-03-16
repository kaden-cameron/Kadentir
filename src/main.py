"""
Main GUI application for face recognition.
Built with PySide6 (Qt for Python).
"""
import sys
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QScrollArea,
    QFileDialog,
    QStatusBar,
    QToolBar,
    QMessageBox,
    QInputDialog,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QThread
from PySide6.QtGui import QImage, QPixmap, QAction

from face_detector import FaceDetector
from face_database import FaceDatabase
from camera_handler import CameraHandler


class CameraStartWorker(QThread):
    """Runs camera.start() in a background thread so the UI does not freeze."""
    finished_signal = Signal(bool)

    def __init__(self, camera_handler: CameraHandler):
        super().__init__()
        self.camera_handler = camera_handler

    def run(self):
        result = self.camera_handler.start()
        self.finished_signal.emit(result)


class FaceItemWidget(QFrame):
    """Widget representing a single saved face in the list."""

    rename_requested = Signal(str, str)

    def __init__(self, name: str, display_name: str, face_path: str):
        super().__init__()
        self.name = name
        self.display_name = display_name
        self.face_path = face_path

        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(64, 64)
        self.thumbnail.setScaledContents(True)
        self.load_thumbnail()
        layout.addWidget(self.thumbnail)

        self.name_label = QLabel(display_name)
        self.name_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.name_label, 1)

        rename_btn = QPushButton("✏️")
        rename_btn.setFixedSize(30, 30)
        rename_btn.clicked.connect(self.on_rename_clicked)
        layout.addWidget(rename_btn)

        self.setLayout(layout)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)

    def load_thumbnail(self):
        """Load and display face thumbnail."""
        if Path(self.face_path).exists():
            image = cv2.imread(self.face_path)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.thumbnail.setPixmap(QPixmap.fromImage(qt_image.copy()))

    def on_rename_clicked(self):
        """Handle rename button click."""
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Face",
            "Enter new name:",
            text=self.display_name,
        )
        if ok and new_name and new_name != self.display_name:
            self.rename_requested.emit(self.name, new_name)
            self.name_label.setText(new_name)
            self.display_name = new_name


class FaceRecognitionApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.detector = FaceDetector()
        self.database = FaceDatabase("data")
        self.camera = CameraHandler()

        self.current_mode = None
        self.current_image = None
        self.current_face_locations = []
        self.current_face_landmarks = None
        self._live_frame_count = 0
        self._camera_start_worker = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_feed)

        self.setup_ui()
        self.load_saved_faces()

    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        left_layout = QVBoxLayout()

        # Toolbar
        toolbar = QToolBar()
        self.live_btn = QPushButton("📹 Live Feed")
        self.live_btn.clicked.connect(self.start_live_feed)
        toolbar.addWidget(self.live_btn)
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        toolbar.addWidget(self.upload_btn)
        self.save_face_btn = QPushButton("💾 Save Face")
        self.save_face_btn.clicked.connect(self.save_current_face)
        self.save_face_btn.setEnabled(False)
        toolbar.addWidget(self.save_face_btn)
        left_layout.addWidget(toolbar)

        # Display label
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.display_label.setText("Click 'Live Feed' or 'Upload Image' to start")
        left_layout.addWidget(self.display_label)
        main_layout.addLayout(left_layout, 3)

        # Right panel
        right_layout = QVBoxLayout()
        search_label = QLabel("Search Faces:")
        right_layout.addWidget(search_label)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to search...")
        self.search_input.textChanged.connect(self.on_search_changed)
        right_layout.addWidget(self.search_input)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumWidth(300)
        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout()
        self.faces_layout.setAlignment(Qt.AlignTop)
        self.faces_container.setLayout(self.faces_layout)
        self.scroll_area.setWidget(self.faces_container)
        right_layout.addWidget(self.scroll_area)
        main_layout.addLayout(right_layout, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def start_live_feed(self):
        """Start or stop live camera feed."""
        if self.current_mode == "live":
            self.timer.stop()
            self.camera.stop()
            self.current_mode = None
            self.live_btn.setText("📹 Live Feed")
            self.display_label.setText("Live feed stopped")
            self.save_face_btn.setEnabled(False)
            self.status_bar.showMessage("Live feed stopped")
            return

        # Start camera in a background thread to avoid freezing the UI (Windows "Not Responding")
        if self._camera_start_worker is not None and self._camera_start_worker.isRunning():
            return
        self.live_btn.setEnabled(False)
        self.status_bar.showMessage("Starting camera...")
        self._camera_start_worker = CameraStartWorker(self.camera)
        self._camera_start_worker.finished_signal.connect(self._on_camera_start_finished)
        self._camera_start_worker.start()

    def _on_camera_start_finished(self, success: bool):
        """Called when background camera start completes."""
        self._camera_start_worker = None
        self.live_btn.setEnabled(True)
        if success:
            self.current_mode = "live"
            self.timer.start(30)
            self.live_btn.setText("⏹️ Stop Feed")
            self.status_bar.showMessage("Live feed started")
        else:
            QMessageBox.warning(self, "Camera Error", "Failed to start camera")
            self.status_bar.showMessage("Ready")

    def update_live_feed(self):
        """Update live feed frame (called by timer)."""
        frame = self.camera.read_frame()
        if frame is None:
            return

        # Run face location detection on half-size frame for speed
        h, w = frame.shape[:2]
        process_scale = 2
        small_w, small_h = max(1, w // process_scale), max(1, h // process_scale)
        small_frame = cv2.resize(frame, (small_w, small_h))
        face_locations_small, _ = self.detector.detect_faces(small_frame)
        # Scale face locations back to full resolution
        face_locations = [
            (t * process_scale, r * process_scale, b * process_scale, l * process_scale)
            for (t, r, b, l) in face_locations_small
        ]

        self.current_image = frame.copy()
        self.current_face_locations = face_locations

        if self.current_face_locations:
            self.detector.draw_face_boxes(frame, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
            # Draw blue face mesh (tessellation + contours) per face using crop so MediaPipe always sees a face
            for face_loc in self.current_face_locations:
                self.detector.draw_face_mesh_on_frame(frame, face_loc, padding=30)
        else:
            self.save_face_btn.setEnabled(False)
            self.current_face_landmarks = None

        # Throttle heavy encoding + DB match to every 5th frame
        self._live_frame_count += 1
        if self.current_face_locations and self._live_frame_count % 5 == 0:
            first_face = self.current_face_locations[0]
            encoding = self.detector.get_face_encoding(self.current_image, first_face)
            if encoding is not None:
                match = self.database.find_closest_match(encoding)
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        self.display_frame(frame)

    def upload_image(self):
        """Upload and process a static image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            return

        if self.current_mode == "live":
            self.timer.stop()
            self.camera.stop()
            self.live_btn.setText("📹 Live Feed")

        self.current_mode = "image"
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image")
            return

        self.current_image = image.copy()
        self.current_face_locations, self.current_face_landmarks = self.detector.detect_faces(image)

        if self.current_face_locations:
            image = self.detector.draw_face_boxes(image, self.current_face_locations)
            self.save_face_btn.setEnabled(True)
            # Draw blue face mesh (tessellation + contours) per face using crop
            for face_loc in self.current_face_locations:
                self.detector.draw_face_mesh_on_frame(image, face_loc, padding=30)
        else:
            self.save_face_btn.setEnabled(False)
            QMessageBox.information(self, "No Faces", "No faces detected in image")

        if self.current_face_locations:
            first_face = self.current_face_locations[0]
            encoding = self.detector.get_face_encoding(self.current_image, first_face)
            if encoding is not None:
                match = self.database.find_closest_match(encoding)
                if match:
                    name, distance = match
                    self.status_bar.showMessage(f"Match: {name} (distance: {distance:.3f})")
                else:
                    self.status_bar.showMessage("No match found")

        self.display_frame(image)

    def save_current_face(self):
        """Save the first detected face to the database."""
        if not self.current_face_locations or self.current_image is None:
            QMessageBox.warning(self, "No Face", "No face detected to save")
            return

        name, ok = QInputDialog.getText(self, "Save Face", "Enter person's name:")
        if not ok or not name.strip():
            return

        first_face = self.current_face_locations[0]
        face_image = self.detector.crop_face(self.current_image, first_face)
        encoding = self.detector.get_face_encoding(self.current_image, first_face)
        if encoding is None:
            QMessageBox.warning(self, "Error", "Failed to generate face encoding")
            return

        if self.database.save_face(name.strip(), face_image, encoding):
            QMessageBox.information(self, "Success", f"Face saved as '{name.strip()}'")
            self.load_saved_faces()
        else:
            QMessageBox.warning(self, "Error", "Failed to save face")

    def load_saved_faces(self, search_query: str = ""):
        """Load and display saved faces in the sidebar."""
        for i in reversed(range(self.faces_layout.count())):
            widget = self.faces_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if search_query.strip():
            faces = self.database.search_faces(search_query)
        else:
            faces = self.database.get_all_faces()

        for face in faces:
            widget = FaceItemWidget(
                face["name"],
                face["display_name"],
                face["face_path"],
            )
            widget.rename_requested.connect(self.on_face_renamed)
            self.faces_layout.addWidget(widget)

    def on_search_changed(self, text: str):
        """Handle search input changes."""
        self.load_saved_faces(text)

    def on_face_renamed(self, old_name: str, new_name: str):
        """Handle face rename request."""
        if self.database.rename_face(old_name, new_name):
            self.status_bar.showMessage(f"Renamed to '{new_name}'")

    def display_frame(self, frame: np.ndarray):
        """Display a frame in the GUI."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        display_width = self.display_label.width()
        display_height = self.display_label.height()
        aspect = w / h
        if display_width / display_height > aspect:
            new_height = display_height
            new_width = int(new_height * aspect)
        else:
            new_width = display_width
            new_height = int(new_width / aspect)
        resized = cv2.resize(rgb_frame, (new_width, new_height))
        h, w, ch = resized.shape
        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image.copy()))

    def closeEvent(self, event):
        """Handle window close - cleanup resources."""
        self.timer.stop()
        self.camera.stop()
        self.detector.cleanup()
        event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
