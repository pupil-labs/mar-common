from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .eye_tracking_source_widget import EyeTrackingSourceWidget
from .hddigital import HDDigitalWidget
from .neon_remote import NeonRemoteWidget
from .neon_usb import NeonLocalWidget


class SourceWidget(QWidget):
    new_device_connected = Signal(object)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        try:
            self.neon_usb_widget = NeonLocalWidget()
            self.tabs.addTab(self.neon_usb_widget, "Neon USB")
        except (ValueError, ModuleNotFoundError):
            pass

        self.neon_remote_widget = NeonRemoteWidget()
        self.tabs.addTab(self.neon_remote_widget, "Neon Remote")

        self.hd_digital_widget = HDDigitalWidget()
        self.tabs.addTab(self.hd_digital_widget, "HD Digital")

        # Connections
        for eye_tracking_source_widget in self.tabs.findChildren(
            EyeTrackingSourceWidget
        ):
            eye_tracking_source_widget.new_device_connected.connect(
                lambda device: self.new_device_connected.emit(device)
            )
            eye_tracking_source_widget.disconnect_requested.connect(
                lambda: self.disconnect_requested.emit()
            )

        layout.addWidget(self.tabs)
        self.setLayout(layout)
