from PySide6.QtCore import QThread, Signal

from pupil_labs.mar_common.eye_tracking_sources.neon_usb import NeonUSB


class DeviceConnectionWorker(QThread):
    success = Signal(object)
    failure = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            device = NeonUSB()
        except Exception:
            self.failure.emit()
            return

        self.success.emit(device)
