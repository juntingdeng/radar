import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import numpy as np
from utils.capon import *
from scipy.interpolate import griddata
# from utils.camera import CameraStream
from utils.cfar import *

class MyApp(QtWidgets.QWidget):
    def __init__(self, radar,
                    collector,
                    live,
                    adc_frames,
                    npeaks,
                    range_ticks_real,
                    range_tick_labels_real,
                    lock=None):
        
        super().__init__()
        self.radar = radar
        self.collector = collector
        self.live = live
        self.frame_idx = 0
        self.adc_frames = adc_frames
        self.npeaks = npeaks
        self.lock = lock

        self.plots = [None]*len(self.radar.angles)
        self.images = [None]*len(self.radar.angles)
        self.scatters = [None]*len(self.radar.angles)
        
        self.BEV = True

        # self.camera = CameraStream()

        angle_pos = np.linspace(0, 181, 9)
        angle_tick = np.linspace(-90, 90, 9).round(1)
        range_tick_labels_real_ = range_tick_labels_real[::-1]
        print(range_tick_labels_real, range_ticks_real)
        angle_ticks = [(angle_pos[i], str(angle_tick[i])) for i in range(angle_pos.shape[0])]
        range_ticks = [(range_ticks_real[i], str(range_tick_labels_real_[i])) for i in range(range_ticks_real.shape[0])]
        print(range_ticks)

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
        
        if self.BEV:
            self.bev_plot = self.plot_widget.addPlot(row=1, col=0, colspan=len(self.radar.angles))
            self.bev_image = pg.ImageItem()
            self.bev_plot.addItem(self.bev_image)
            self.bev_plot.setTitle("BEV (Bird's Eye View)")
            self.bev_plot.getAxis('bottom').setLabel("X (meters)")
            self.bev_plot.getAxis('left').setLabel("Y (meters)")
            self.bev_plot.invertY(False)
        
        # cam_plot = self.plot_widget.addPlot(row=1, col=0, colspan=len(self.radar.angles))
        # self.cam_frame = pg.ImageItem()
        # cam_plot.addItem(self.cam_frame)
        # self.cam_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 255, 0, 200))  # white dots
        # cam_plot.addItem(self.cam_scatter)
            
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def update(self):
        adc_data = None
        cam_frame = None
        if self.live:
            with self.lock:
                
                if self.collector.visualize_buffer is not None and self.collector.visualize_buffer.shape[0] > self.radar.samples_per_frame:
                    # cam_frame = self.cam.read()

                    adc_raw = self.collector.visualize_buffer[: self.radar.samples_per_frame]
                    self.collector.visualize_buffer = self.collector.visualize_buffer[self.radar.samples_per_frame: ]
                    adc_raw = adc_raw.reshape(self.radar.num_chirps, self.radar.num_rx, self.radar.num_adc_samples)
                    adc_data = np.concatenate([adc_raw[i::self.radar.num_tx, :, :] for i in range(self.radar.num_tx)], axis=-2)
                    print(f'One frame received. Caputred data shape:{adc_data.shape}')

        else:
            adc_data = self.adc_frames[self.frame_idx]
            self.frame_idx += 1
            if self.frame_idx >= self.adc_frames.shape[0]:
                self.timer.stop()
        
        if adc_data is not None: # and cam_frame is not None:
            ## Perform 2d FFT for azimuth-elevation estimation
            print(f'adc_data shape: {adc_data.shape}')
            print(f'self.radar.num_rx: {self.radar.num_rx}, self.radar.num_tx: {self.radar.num_tx}, self.radar.num_chirps: {self.radar.num_chirps}, self.radar.num_adc_samples: {self.radar.num_adc_samples}')
            range_cube = np.fft.fft(adc_data, axis=2).transpose(2, 1, 0) #[range, channel, doppler]
            print(f'Range cube shape: {range_cube.shape}')
            range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2) #[range, channel, doppler]  
            
            range_azimuths = []
            detecteds = []
            # fig, ax = plt.subplots(len(num_ant), 2, figsize=(9, 10))
            for i, (n, angle) in enumerate(zip(self.radar.num_ant, self.radar.angles)):
                print(f'Processing {angle} with {n} antennas, index: {i}')
                if angle == 'Elevation':
                    range_doppler_mean = range_doppler.reshape(range_doppler.shape[0], self.radar.num_rx, self.radar.num_tx, range_doppler.shape[-1])
                    range_doppler_mean = np.mean(range_doppler_mean, axis=1)
                else:
                    range_doppler_mean = range_doppler

                ## Beamformer
                n_range_bins = range_doppler_mean.shape[0]
                n_angles = self.radar.steering_vectors[i].shape[0]
                print(f'steering vectors shape: {self.radar.steering_vectors[i].shape}')
                range_azimuth = np.zeros((n_range_bins, n_angles), dtype=np.complex64)
                for range_i in range(range_doppler_mean.shape[0]):
                    range_azimuth[range_i,:] = aoa_capon(range_doppler_mean[range_i, ...], self.radar.steering_vectors[i])
                range_azimuth = np.flipud(range_azimuth)
                range_azimuth = 10*np.log10(np.abs(range_azimuth))
                range_azimuth = range_azimuth[range_azimuth.shape[0]//2:, :][::2]
                
                self.images[i].setImage(np.fliplr(range_azimuth.transpose()), 
                                    lut=pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256))
                
                range_azimuths.append(range_azimuth)
                detected, range_dect, angle_dect = cfar(guard_len=2, train_len=3, p_fa=1e-8, range_azimuth=range_azimuth) 
                detecteds.append(detected)

                # ## Select top self.npeaks as detected points
                # indices = np.argpartition(range_azimuth[:].flatten(), -self.npeaks)[-self.npeaks: ]
                # peak_idx = np.unravel_index(indices, range_azimuth.shape)
                # range_bins, doppler_bins = peak_idx
                # points = []
                # for r, d in zip(range_bins, doppler_bins):
                #     points.append({'pos':(d, range_azimuth.shape[0]//2 - (r//2))})
                
                ## CAFR to select detected points
                points = []
                for y in range(detected.shape[0]):
                    for x in range(detected.shape[1]):
                        if detected[y][x] == 1:
                            points.append({'pos':(x, range_azimuth.shape[0]-y)})
                self.scatters[i].setData(points)

                if angle == 'Azimuth':
                    if self.BEV:
                        angle_bins_full = np.linspace(-np.pi/2, np.pi/2, n_angles)
                        angle_mask = (angle_bins_full >= -np.pi/4) & (angle_bins_full <= np.pi/4)
                        angle_bins = angle_bins_full[angle_mask] # no down-sampling for better resolution
                        # range_az = range_azimuth[::2, :][:, angle_mask]
                        range_az = range_azimuth[:, angle_mask]
                        range_az = np.flipud(range_az)

                        n_range_bins = range_az.shape[0]
                        range_bins = np.linspace(0, self.radar.max_range / 2, n_range_bins)
                        range_max = self.radar.max_range / 2
                        # Create polar coordinate grid (rsin(theta), rcos(theta))
                        r_grid, a_grid = np.meshgrid(range_bins, angle_bins, indexing='ij')
                        x = r_grid * np.sin(a_grid)
                        y = r_grid * np.cos(a_grid)

                        # Interpolate to a uniform Cartesian grid (only +- 45 degrees)
                        xlin = np.linspace(-range_max / np.sqrt(2), range_max / np.sqrt(2), 512)
                        ylin = np.linspace(0, range_max, 512)
                        x_grid, y_grid = np.meshgrid(xlin, ylin)

                        points = np.column_stack((x.flatten(), y.flatten()))
                        values = range_az.flatten()

                        # Interpolation
                        bev_cartesian = griddata(points, values, (x_grid, y_grid), method='linear', fill_value=np.nan)
                        # bev_cartesian = np.flipud(bev_cartesian)

                        lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
                        if lut.shape[1] == 3:
                            alpha = 255 * np.ones((lut.shape[0], 1), dtype=np.uint8)
                            lut = np.hstack([lut, alpha])
                        white = np.array([[255, 255, 255, 255]], dtype=np.uint8)
                        lut_with_white = np.vstack([lut, white])

                        self.bev_image.setImage(
                            bev_cartesian.T,
                            lut=lut_with_white
                        )
                            # levels=(min_dB, max_dB),
                        xlin = np.array(xlin)
                        ylin = np.array(ylin)
                        self.bev_image.setRect(QtCore.QRectF(xlin.min() * 0.5, ylin.min() * 0.5, np.ptp(xlin) * 0.5, np.ptp(ylin) * 0.5))

            # detected_radar_points = radar_points(detecteds=detecteds, angles=self.radar.angles, radar=self.radar) 
            
            # # Visualization (optional)
            # img_w, img_h = cam_frame.shape[1], cam_frame.shape[0]
            # print(img_w, img_h)
            # detected_radar_points = np.array(detected_radar_points)
            # # Project the radar points
            # projected_points_cal = project_points(detected_radar_points, K_calibrated)
            # depth_mask = np.zeros((img_h, img_w))
            # print(depth_mask.shape)
            # for i in range(projected_points_cal.shape[0]):
            #     x, y, d = projected_points_cal[i]
            #     if 0<=round(y)<=img_h and 0<=round(x)<=img_w:
            #         depth_mask[round(y)][round(x)] = d

            # self.cam_scatter.setData(projected_points_cal[:, :2])

            QtWidgets.QApplication.processEvents()