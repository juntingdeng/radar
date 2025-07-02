import h5py
import collections
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import threading
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\..\\..\\"))
from processing.process import radar, dataset
from radar.collect import _read_data_packet
from radar.collect import *
from util import *

# dataset = dataset.AWR1843BoostDataset

num_rx = 4
num_tx = 4
start_freq = 77 # GHz
idle_time = 10 # usec
adc_start_time = 6 # usec
ramp_end_time = 120 # usec, ramp_end_time = adc_start_time + adc_sample_time(per chirp) + extral_time
chirp_slope =  29.982 # MHz/usec
num_adc_samples = 1024
adc_sample_rate = 10 # Msps
num_frames = 256
num_chirps = 255
adc_sample_time = num_adc_samples/adc_sample_rate # usec
chirps = 4


frame_duration = (ramp_end_time+idle_time)*num_chirps*1e-6 # sec
frame_act_ratio = adc_sample_time/(ramp_end_time+idle_time)
frame_start_timestamps = [i*frame_duration+adc_start_time*1e-6 for i in range(num_frames)] # usec
frame_sample_time = [[i*frame_duration, i*frame_duration+adc_sample_time] for i in range(num_frames)] # [frame_start, frame_end] usec

print('-------Config--------')
print(f"Number of Rx Antennas: {num_rx}")
print(f"Number of Tx Antennas: {num_tx}")
print(f"Start Frequency: {start_freq} GHz")
print(f"Idle Time: {idle_time} usec")
print(f"ADC Start Time Time: {adc_start_time} usec")
print(f'ADC sample time: {adc_sample_time} usec')
print(f"Ramp End Time: {ramp_end_time} usec") # this should really be called ramp to end? or maybe just chirptime?
print(f"Chirp Slope: {chirp_slope} MHz/usec")
print(f"Number of ADC Samples: {num_adc_samples}")
print(f"ADC sample rate: {adc_sample_rate} Msps")
print(f'Frame Duration:{frame_duration:.4f} sec')
print(f'FPS:{int(1/(frame_duration))}')
print(f'Frame Active Ratio:{frame_act_ratio:.4f}')

num_chirps = num_chirps*num_tx
samples_per_frame = num_adc_samples*num_chirps*num_rx
bytes_per_frame = 2*samples_per_frame
packets_per_frame = bytes_per_frame//(728*2)
print(f'Samples per frames: {samples_per_frame}')
print(f'Bytes per frame: {bytes_per_frame}')
print(f'Packets per frame: {packets_per_frame}')


print('-------Resolution--------')
# range_resolution, bandwidth = compute_range_resolution(adc_sample_rate, num_adc_samples, chirp_slope)
bandwidth = chirp_slope*ramp_end_time*1e6
range_resolution = LIGHT_SPEED/(2*bandwidth)
max_range = range_resolution * num_adc_samples
print(f'range_resolution: {range_resolution}, max_range: {max_range}')

# num_chirps = adc_data.shape[0]
chirp_interval = (ramp_end_time + idle_time) * 1e-6 # usec

doppler_resolution = compute_doppler_resolution(num_chirps, bandwidth, chirp_interval, num_tx)
max_doppler = doppler_resolution * num_chirps / 2

print(f'doppler_resolution: {doppler_resolution}, doppler_resolution: {max_doppler}')

ranges = np.arange(0, max_range + range_resolution, range_resolution)
range_ticks = np.arange(0, len(ranges)/4, len(ranges)//40)
range_tick_labels = ranges[::len(ranges)//10].round(2)
print(ranges.shape, range_ticks.shape, range_tick_labels.shape)

print('-------Packet data--------')
lock = threading.Lock()
visualize_buffer = None
def collector():
    global visualize_buffer
    with open("config.json") as f:
        cfg = json.load(f)

    _config_socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    _config_socket.bind((cfg["static_ip"], cfg["config_port"]))

    data_socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    data_socket.bind((cfg["static_ip"], cfg["data_port"]))
    data_socket.settimeout(10)

    # Instruct the mmWave Studio Lua server to instruct the radar to start
    # collecting data.
    with open(cfg["msgfile"], 'w') as f:
        f.write("start")
        print("written start.")

    with tb.open_file(datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + ".h5", 
                    mode='w', title='Packet file') as h5file:
        dataset = DataCollector(h5file)
        try:
            while True:
                packet_num, byte_count, packet_data = _read_data_packet(data_socket)
                dataset.write_packet(packet_num, byte_count, packet_data)
                if dataset.chunk_packets >= PACKET_BUFSIZE:
                    dataset.flush()

                # if not lock.locked(): 
                with lock:
                    # print(f'Inside collector lock')
                    if visualize_buffer is not None:
                        # if visualize_buffer.shape[0] % 728 != 0:
                        #     print(visualize_buffer.shape, packet_data.shape)
                        #     break
                        visualize_buffer = np.concatenate([visualize_buffer, packet_data.reshape(-1)], axis=0)
                        
                    else:
                        visualize_buffer = packet_data.reshape(-1)

        except Exception as e:
            print("Radar data collection failed. Was the radar shut down?")
            print(f'Capture Exception: {e}')
        except KeyboardInterrupt:
            print(
                "Received KeyboardInterrupt, stopping.\n"
                "Do not press Ctrl+C again!")
            with open(cfg["msgfile"], 'w') as f:
                f.write("stop")
        finally:
            dataset.flush()


angles = ['Elevation', 'Azimuth']
num_ant = [num_tx, num_rx*num_tx]

ranges_real = np.arange(0, max_range/2 + range_resolution, range_resolution)
range_ticks_real = np.arange(0, len(ranges_real)/2, len(ranges_real)//20)

range_tick_labels_real = ranges_real[::len(ranges_real)//10].round(2)[::-1]
print('ranges_real', ranges_real.shape)

# range_ticks_real = range_ticks[range_ticks.shape[0]//2: ]
# range_tick_labels_real = range_tick_labels[::-1][range_tick_labels.shape[0]//2: ]
print('range_ticks_real', range_ticks_real.shape)
print('range_tick_labels_real', range_tick_labels_real.shape)

steering_vectors = []
for i, (n, angle) in enumerate(zip(num_ant, angles)):
    steering_vectors.append(compute_steering_vector(num_ant=n, angle_res=1.0, angle_rng=90))

plots = [[None]*2 for _ in range(2)]
images = [[None]*2 for _ in range(2)]
scatters = [[None]*2 for _ in range(2)]
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Range-Doppler Viewer")
win.resize(800, 500)
win.show()

angle_pos = np.linspace(0, 181, 9)
angle_tick = np.linspace(-90, 90, 9).round(1)
range_tick_labels_real_ = range_tick_labels_real[::-1]
angle_ticks = [(angle_pos[i], str(angle_tick[i])) for i in range(angle_pos.shape[0])]
range_ticks = [(range_ticks_real[i], str(range_tick_labels_real_[i])) for i in range(range_ticks_real.shape[0])]
print(f'angle_ticks: {angle_ticks}')
print(f'range_ticks: {range_ticks}')
for i in range(1):
    for j in range(2):
        p = win.addPlot(row=i, col=j)
        img = pg.ImageItem()
        p.addItem(img)
        scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 255, 0, 200))  # white dots
        p.addItem(scatter)
        p.setTitle(f"{angles[j]}")
    
        p.getAxis('bottom').setTicks([angle_ticks])
        p.getAxis('bottom').setLabel(f"{angles[j]} (degrees)")
        p.getAxis('left').setTicks([range_ticks])
        p.getAxis('left').setLabel("Range (meters)")

        plots[i][j] = p
        images[i][j] = img
        scatters[i][j] = scatter
cmap = pg.colormap.get('viridis')  # or 'viridis', 'hot', etc.
lut = cmap.getLookupTable(0.0, 1.0, 256)


def update():
    global visualize_buffer
    adc_raw = None
    # if not lock.locked():
    #     with lock:
    if visualize_buffer is not None and visualize_buffer.shape[0] > samples_per_frame:
        adc_raw = visualize_buffer[: samples_per_frame]
        visualize_buffer = visualize_buffer[samples_per_frame: ]
        
    if adc_raw is not None:
        print("One frame received.")
        adc_raw = adc_raw.reshape(num_chirps, num_rx, num_adc_samples)
        adc_data = np.concatenate([adc_raw[i::num_tx, :, :] for i in range(num_tx)], axis=-2)
        print(f'Caputred data shape:{adc_data.shape}')
        
        ## Perform 2d FFT for azimuth-elevation estimation
        range_cube = np.fft.fft(adc_data, axis=2).transpose(2, 1, 0) #[range, channel, doppler]
        range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2) #[range, channel, doppler]
        
        # fig, ax = plt.subplots(len(num_ant), 2, figsize=(9, 10))
        for i, (n, angle) in enumerate(zip(num_ant, angles)):
            if angle == 'Elevation':
                range_doppler_mean = range_doppler.reshape(range_doppler.shape[0], num_rx, num_tx, range_doppler.shape[-1])
                range_doppler_mean = np.mean(range_doppler_mean, axis=1)
                print(range_doppler_mean.shape)
            else:
                range_doppler_mean = range_doppler

            ## Beamformer
            n_range_bins = range_doppler_mean.shape[0]
            n_angles = steering_vectors[i].shape[0]
            range_azimuth = np.zeros((n_range_bins, n_angles), dtype=np.complex64)
            for range_i in range(range_doppler_mean.shape[0]):
                range_azimuth[range_i,:] = aoa_capon(range_doppler_mean[range_i, ...], steering_vectors[i])
            range_azimuth = np.flipud(range_azimuth)
            range_azimuth = 10*np.log10(np.abs(range_azimuth))
            range_azimuth = range_azimuth[range_azimuth.shape[0]//2:, :]
            images[0][i].setImage(np.fliplr(range_azimuth[::2, :].transpose()), lut=lut)

            npoints = 16
            indices = np.argpartition(range_azimuth[:].flatten(), -npoints)[-npoints: ]
            peak_idx = np.unravel_index(indices, range_azimuth.shape)
            range_bins, doppler_bins = peak_idx
            points = []
            for r, d in zip(range_bins, doppler_bins):
                points.append({'pos':(d, range_azimuth.shape[0]//2 - (r//2))})
            scatters[0][i].setData(points)
            
        QtWidgets.QApplication.processEvents()

thread = threading.Thread(target=collector, daemon=True)
thread.start()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

# Start Qt event loop
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()