import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import gi
# gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# === GStreamer Camera Reader ===
class CameraStream:
    def __init__(self):
        pipeline_str = """
        qtiqmmfsrc camera=0 !
        video/x-raw, width=1280, height=720, framerate=30/1 !
        videoconvert !
        video/x-raw, format=BGR !
        appsink name=sink emit-signals=true max-buffers=1 drop=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_new_sample)
        self.sample = None
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape((height, width, 3))
        buf.unmap(mapinfo)
        self.sample = frame.copy()
        return Gst.FlowReturn.OK

    def read(self):
        return self.sample.copy() if self.sample is not None else None