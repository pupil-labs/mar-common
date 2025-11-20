from pupil_labs.mar_common.eye_tracking_sources.neon_usb import NeonUSB

neon_usb = NeonUSB()
neon_usb.get_sample()
neon_usb.close()
