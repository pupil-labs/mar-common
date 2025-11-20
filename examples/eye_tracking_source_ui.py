import sys

import numpy as np
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtGui import QPainter, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingSource
from pupil_labs.mar_common.ui.eye_tracking_source import SourceWidget
from pupil_labs.mar_common.ui.scaled_image_view import ScaledImageView


class EyeTrackingOverlay(ScaledImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gaze = None

    def set_gaze(self, gaze):
        if gaze is None:
            self.gaze = None
        else:
            assert self.image is not None
            img_size = self.image.size().toTuple()
            overlay_size = self.size().toTuple()
            scale_factor = (
                overlay_size[0] / img_size[0],
                overlay_size[1] / img_size[1],
            )
            gaze = np.array(gaze)
            self.gaze = (gaze * scale_factor).astype(int)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.gaze is not None:
            with QPainter(self) as painter:
                radius = 10
                painter.setBrush(Qt.red)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    QPoint(*self.gaze),
                    radius,
                    radius,
                )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracking Source Selection")

        self.device: EyeTrackingSource | None = None

        selection_widget = SourceWidget()
        layout = QVBoxLayout()
        layout.addWidget(selection_widget)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.preview = EyeTrackingOverlay()
        self.preview.setMinimumSize(640, 480)
        self.preview.setWindowTitle("Eye Tracking Overlay")
        self.preview.show()

        selection_widget.new_device_connected.connect(self.on_new_device_connected)
        selection_widget.disconnect_requested.connect(self.on_disconnect_requested)

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(int(1000 / 30))
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()

    def on_new_device_connected(self, eye_tracking_source):
        self.device = eye_tracking_source

    def on_disconnect_requested(self):
        if self.device is not None:
            self.device.close()
            self.device = None
        self.preview.set_image(None)
        self.preview.set_gaze(None)

    def poll(self):
        if self.device is not None:
            et_data = self.device.get_sample()
            self.preview.set_image(et_data.scene)
            if et_data.gaze is not None:
                self.preview.set_gaze(et_data.gaze)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
