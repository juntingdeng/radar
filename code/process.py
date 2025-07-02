import h5py
import collections
import numpy as np
# import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import threading
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\..\\..\\"))
# from processing.process import radar, dataset
from utils.parse_config import *
from utils.collection import *

class MyApp(QtWidgets.QWidget):
    def __init__(self, radar,
                    visualize_buffer,
                    live,
                    adc_frames,
                    npeaks,
                    range_ticks_real,
                    range_tick_labels_real):
        
        super().__init__()
        self.radar = radar
        self.visualize_buffer = visualize_buffer
        self.live = live
        self.frame_idx = 0
        self.adc_frames = adc_frames
        self.npeaks = npeaks

        self.plots = [None]*len(self.radar.angles)
        self.images = [None]*len(self.radar.angles)
        self.scatters = [None]*len(self.radar.angles)

        angle_pos = np.linspace(0, 181, 9)
        angle_tick = np.linspace(-90, 90, 9).round(1)
        range_tick_labels_real_ = range_tick_labels_real[::-1]
        angle_ticks = [(angle_pos[i], str(angle_tick[i])) for i in range(angle_pos.shape[0])]
        range_ticks = [(range_ticks_real[i], str(range_tick_labels_real_[i])) for i in range(range_ticks_real.shape[0])]

        layout = QtWidgets.QVBoxLayout(self)
        # Create GraphicsLayoutWidget
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)

        for i in range(len(self.radar.angles)):
            p = self.plot_widget.addPlot(row=0, col=i)
            img = pg.ImageItem()
            p.addItem(img)
            scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 255, 0, 200))  # white dots
            p.addItem(scatter)
            p.setTitle(f"{self.radar.angles[i]}")

            p.getAxis('bottom').setTicks([angle_ticks])
            p.getAxis('bottom').setLabel(f"{self.radar.angles[i]} (degrees)")
            p.getAxis('left').setTicks([range_ticks])
            p.getAxis('left').setLabel("Range (meters)")

            self.plots[i] = p
            self.images[i] = img
            self.scatters[i] = scatter

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def update(self):
        adc_data = None

        if self.live:
            if visualize_buffer is not None and visualize_buffer.shape[0] > self.radar.samples_per_frame:
                adc_raw = visualize_buffer[: self.radar.samples_per_frame]
                visualize_buffer = visualize_buffer[self.radar.samples_per_frame: ]
                print("One frame received.")
                adc_raw = adc_raw.reshape(self.radar.num_chirps, self.radar.num_rx, self.radar.num_adc_samples)
                adc_data = np.concatenate([adc_raw[i::self.radar.num_tx, :, :] for i in range(self.radar.num_tx)], axis=-2)
                print(f'Caputred data shape:{adc_data.shape}')
        else:
            adc_data = self.adc_frames[self.frame_idx]
            self.frame_idx += 1
            if self.frame_idx >= self.adc_frames.shape[0]:
                self.timer.stop()
        
        if adc_data is not None:
            ## Perform 2d FFT for azimuth-elevation estimation
            range_cube = np.fft.fft(adc_data, axis=2).transpose(2, 1, 0) #[range, channel, doppler]
            range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2) #[range, channel, doppler]
            
            # fig, ax = plt.subplots(len(num_ant), 2, figsize=(9, 10))
            for i, (n, angle) in enumerate(zip(self.radar.num_ant, self.radar.angles)):
                if angle == 'Elevation':
                    range_doppler_mean = range_doppler.reshape(range_doppler.shape[0], self.radar.num_rx, self.radar.num_tx, range_doppler.shape[-1])
                    range_doppler_mean = np.mean(range_doppler_mean, axis=1)
                else:
                    range_doppler_mean = range_doppler

                ## Beamformer
                n_range_bins = range_doppler_mean.shape[0]
                n_angles = self.radar.steering_vectors[i].shape[0]
                range_azimuth = np.zeros((n_range_bins, n_angles), dtype=np.complex64)
                for range_i in range(range_doppler_mean.shape[0]):
                    range_azimuth[range_i,:] = aoa_capon(range_doppler_mean[range_i, ...], self.radar.steering_vectors[i])
                range_azimuth = np.flipud(range_azimuth)
                range_azimuth = 10*np.log10(np.abs(range_azimuth))
                range_azimuth = range_azimuth[range_azimuth.shape[0]//2:, :]
                self.images[i].setImage(np.fliplr(range_azimuth[::2, :].transpose()), 
                                    lut=pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256))

                indices = np.argpartition(range_azimuth[:].flatten(), -self.npeaks)[-self.npeaks: ]
                peak_idx = np.unravel_index(indices, range_azimuth.shape)
                range_bins, doppler_bins = peak_idx
                points = []
                for r, d in zip(range_bins, doppler_bins):
                    points.append({'pos':(d, range_azimuth.shape[0]//2 - (r//2))})
                self.scatters[i].setData(points)
                
            QtWidgets.QApplication.processEvents()


def main(args):
    radar = radarConfig()
    range_ticks_real, range_tick_labels_real = radar.parse_radar(cfg_file=args.cfg_file)

    ## if Live: data capturing while processing, else: read captured data from .h5 file
    live = args.live
    visualize_buffer = None
    adc_frames = None
    npeaks = args.npeaks
    if live:
        lock = threading.Lock()
        thread = threading.Thread(target=collector, daemon=True)
        thread.start()
    else:
        allkeys = collections.defaultdict(list)
        with h5py.File(args.data_file, 'r') as f:
            packet = f['scan']['packet']
            packet_num = packet['packet_num']
            packet_t = packet['t']
            byte_count = packet['byte_count']
            packet_data = np.array(packet['packet_data'], dtype=np.int16)

        adc_raw = packet_data.reshape(-1)
        print(f'packet_data shape: {packet_data.shape}')
        print(f'adc_raw shape:{adc_raw.shape}')

        nframes = adc_raw.shape[0] // (radar.num_adc_samples*radar.num_chirps*radar.num_rx)
        assert adc_raw.shape[0] == radar.num_adc_samples*radar.num_chirps*radar.num_rx*radar.num_frames or 0 < nframes <radar.num_frames
        print(f'{nframes} received')
        adc_raw = adc_raw[ :radar.num_adc_samples*radar.num_chirps*radar.num_rx*nframes]
        adc_frames = adc_raw.reshape(nframes, radar.num_chirps, radar.num_rx, radar.num_adc_samples)
        adc_tx = [adc_frames[:, i::radar.num_tx, :, :] for i in range(radar.num_tx)]
        adc_frames = np.concatenate(adc_tx, axis=-2)
        print(f'Caputred data shape:{adc_frames.shape}')

    app = QtWidgets.QApplication([])
    win = MyApp(radar=radar,
                visualize_buffer=visualize_buffer,
                live=live,
                adc_frames=adc_frames,
                npeaks=npeaks,
                range_ticks_real=range_ticks_real,
                range_tick_labels_real=range_tick_labels_real)

    win.resize(800, 500)
    win.show()
    app.exec_()

def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--cfg_file', default='./mmWaveStudio/startup_capture.lua')
    args.add_argument('--data_file', default=None, type=str)
    args.add_argument('--live', action='store_true')
    args.add_argument('--npeaks', default=16, type=int)
    return args.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main(args)