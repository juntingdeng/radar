import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
from matplotlib.backends.backend_agg import FigureCanvas
import os
from scipy.ndimage import convolve1d, convolve

from utils.parse_config import *
from utils.cfar import *
from utils.capon import aoa_capon, aoa_capon_nonuniform, compute_steering_vector, compute_steering_vector_nonuniform
sys.path.insert(0, '..')

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

frame_duration = (ramp_end_time+idle_time)*num_chirps*1e-6 # sec
frame_act_ratio = adc_sample_time/(ramp_end_time+idle_time)
frame_start_timestamps = [i*frame_duration+adc_start_time*1e-6 for i in range(num_frames)] # usec
frame_sample_time = [[i*frame_duration, i*frame_duration+adc_sample_time] for i in range(num_frames)] # [frame_start, frame_end] usec

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

cap = 7
paths = [f'./rawData/cap{cap}/data_Raw_{i}.bin' for i in range(2)]
# path1 = f'D:/rawData/cap7/data_Raw_1.bin'

rad_range = np.arange(start=0, stop=np.pi, step=np.pi/(num_adc_samples))
rad_doppler = np.arange(start=-np.pi/2, stop=np.pi/2, step=np.pi/(num_chirps//num_tx))

adc_raw = []
for path in paths:
    adc_raw.append(np.fromfile(path, dtype=np.int16))
adc_raw = np.concatenate(adc_raw, axis=0)
print(f'adc_raw shape:{adc_raw.shape}')

nframes = adc_raw.shape[0] // (num_adc_samples*num_chirps*num_rx)
assert adc_raw.shape[0] == num_adc_samples*num_chirps*num_rx*num_frames or 0 < nframes <num_frames
print(f'{nframes} received')
adc_raw = adc_raw[ :num_adc_samples*num_chirps*num_rx*nframes]

adc_frames = adc_raw.reshape(nframes, num_chirps, num_rx, num_adc_samples)
adc_tx = []
for i in range(num_tx):
    adc_tx.append(adc_frames[:, i::num_tx, :, :])
    print(f'tx{i}:{adc_tx[-1].shape}')
adc_frames = np.concatenate(adc_tx, axis=-2)
print(f'virtual:{adc_frames.shape}')

adc_frames = adc_frames/np.max(adc_frames)
adc_data = adc_frames[22]
print(f'adc_data shape:{adc_data.shape}')

LIGHT_SPEED = 299792458 # m/s
# range_resolution, bandwidth = compute_range_resolution(adc_sample_rate, num_adc_samples, chirp_slope)
bandwidth = chirp_slope*ramp_end_time*1e6
range_resolution = LIGHT_SPEED/(2*bandwidth)
max_range = range_resolution * num_adc_samples
print(f'range_resolution: {range_resolution}, max_range: {max_range}')

num_chirps = adc_data.shape[0]
chirp_interval = (ramp_end_time + idle_time) * 1e-6 # usec

doppler_resolution = compute_doppler_resolution(num_chirps, bandwidth, chirp_interval, num_tx)
max_doppler = doppler_resolution * num_chirps / 2

print(f'doppler_resolution: {doppler_resolution}, doppler_resolution: {max_doppler}')

ranges = np.arange(0, max_range + range_resolution, range_resolution)
range_ticks = np.arange(0, len(ranges)/4, len(ranges)//40)
range_tick_labels = ranges[::len(ranges)//10].round(2)
print(ranges.shape, range_ticks.shape, range_tick_labels.shape)

ranges_real = np.arange(0, max_range/2 + range_resolution, range_resolution)
range_ticks_real = np.arange(0, len(ranges_real)/2, len(ranges_real)//20)
range_tick_labels_real = ranges_real[::len(ranges_real)//10].round(2)[::-1]

## FFT
range_cube = np.fft.fft(adc_data, axis=2).transpose(2, 1, 0) #[range, channel, doppler]
range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2) #[range, channel, doppler]

## Beamformer
steering_vector = compute_steering_vector(num_ant=num_tx*num_rx, angle_res=1.0, angle_rng=90)
print('steering_vector shape:', steering_vector.shape)

n_range_bins = range_doppler.shape[0]
n_angles = steering_vector.shape[0]
range_azimuth = np.zeros((n_range_bins, n_angles), dtype=np.complex64)
for i in range(range_doppler.shape[0]):
    range_azimuth[i,:] = aoa_capon(range_doppler[i, ...], steering_vector)
range_azimuth = np.flipud(range_azimuth)[range_azimuth.shape[0]//2:, :][::2]
range_azimuth = 10*np.log10(np.abs(range_azimuth))
range_azimuth = np.fliplr(range_azimuth)
print('range_azimuth mean: ', np.mean(range_azimuth))

fig, ax = plt.subplots(1, 3, figsize=(9, 5))
for i in range(2):
    ax[i].imshow(range_azimuth)
    ax[i].set_title("Range Azimuth Power Spectrum")
    ax[i].set_xlabel("Azimuth Angle (degrees)")
    ax[i].set_ylabel("Range (meters)")
    ax[i].set_yticks(range_ticks_real, range_tick_labels_real)
    ax[i].set_xticks(np.linspace(0, 181, 9), np.linspace(-90, 90, 9).round(1))
  
guard_len = 2
train_len = 3
p_fa = 0.0000001 # Probability of False 
detected, range_dect, angle_dect = cfar(guard_len=guard_len, 
                                        train_len=train_len, 
                                        p_fa=p_fa, 
                                        range_azimuth=range_azimuth)

for i in range(detected.shape[0]):
    for j in range(detected.shape[1]):
        if detected[i][j] == 1:
            ax[1].scatter(j, i)

import pyvista as pv
## 3d Beamformer
positions = np.array([
[0.0,   0.0], [0.5,   0.0], [1.0,   0.0], [1.5,   0.0],
[1.0,   0.8], [1.5,   0.8], [2.0,   0.8], [2.5,   0.8],
[2.0,   0.0], [2.5,   0.0], [3.0,   0.0], [3.5,   0.0],
[4.0,   0.0], [4.5,   0.0], [5.0,   0.0], [5.5,   0.0],
])
angle_rng_azi = 90
angle_rng_ele = 10
angle_res = 1
steering_vector = compute_steering_vector_nonuniform(positions=positions, angle_ress=[angle_res], angle_rngs=[angle_rng_azi, angle_rng_ele])
print('steering_vector shape (ranges, azimuths, elevationss):', steering_vector.shape)

n_range_bins = range_doppler.shape[0]
n_angles_azi, n_angles_ele = steering_vector.shape[1], steering_vector.shape[2]
# range_azimuth = np.zeros((n_range_bins, n_angles), dtype=np.complex64)
range_azimuth = np.zeros((n_range_bins, n_angles_azi, n_angles_ele))
for range_i in range(range_doppler.shape[0]):
    range_azimuth[range_i,:] = aoa_capon_nonuniform(range_doppler[range_i, ...], steering_vector)
range_azimuth = np.flipud(range_azimuth)
range_azimuth = 10*np.log10(np.abs(range_azimuth))
range_azimuth = range_azimuth[range_azimuth.shape[0]//2:, :, :][::2, :, :]
range_azimuth = np.flip(range_azimuth, axis=1)

# CFAR
ax[2].imshow(range_azimuth[:, :, range_azimuth.shape[-1]//2])
ax[2].set_title(f"Range-Azimuth - Cube X-Y Plane")
ax[2].set_xlabel(f"Azimuth (degrees)")
ax[2].set_ylabel("Range (meters)")
ax[2].set_yticks(range_ticks_real, range_tick_labels_real)
ax[2].set_xticks(np.linspace(0, 2*(angle_rng_azi//angle_res)+1, 9), np.linspace(-angle_rng_azi, angle_rng_azi, 9).round(1))

grid = pv.ImageData()
grid.dimensions = np.array(range_azimuth.shape) + 1
print(grid.dimensions)
grid.origin = (0, 0, 0)  # set origin
grid.spacing = (1, 1, 1)  # set spacing if needed
grid.cell_data["amplitude"] = np.abs(range_azimuth).flatten(order="F")

plotter = pv.Plotter()
plotter.add_volume(grid, cmap="viridis", opacity="sigmoid")
plotter.add_mesh(grid.outline(), color="black")  # Border/outline
plotter.show_grid()  # adds grid lines and axes
plotter.show_axes()  # shows 3D orientation axes in the corner
plotter.show()

p_fa = 1e-16
detected = cfar_3d(guard_len=(2,2,1), train_len=(4, 4, 3), p_fa=p_fa, power_cube=range_azimuth)
print(f'n detected: {np.sum(detected)}')

# convert to radians
radar_points = []
angles = []
# convert to radians
for angle_rng, angle_res in zip([angle_rng_azi, angle_rng_ele], [angle_res, angle_res]):
    angle_rng = angle_rng*np.pi/180
    angle_res = angle_res*np.pi/180

    angles.append(np.arange(start=-angle_rng, stop=angle_rng+angle_res, step=angle_res))

ranges = np.arange(start=0, stop=range_resolution*num_adc_samples//2, step=range_resolution)
for ri in range(detected.shape[0]):
    rang = ranges[ri]
    for ai in range(detected.shape[1]):
        azim = angles[0][ai]
        for ei in range(detected.shape[2]):
            elev = angles[1][ei]
            if detected[ri][ai][ei] == True:

                x = rang * np.cos(elev) * np.cos(azim)
                y = rang * np.cos(elev) * np.sin(azim)
                z = rang * np.sin(elev)
                radar_points.append([y, -z, x])

print(f'num detected: {np.sum(detected)}/{np.prod(detected.shape)}')
# Visualization (optional)
img = Image.open(f'./rawData/cap{cap}/frames/frame_570.png')
plt.figure(figsize=(8, 5))
plt.imshow(np.array(img))

img = np.array(img)
img_w, img_h = img.shape[1], img.shape[0]
print('img size: ', img_w, img_h)
radar_points_new = np.array(radar_points)
print(f'radar_points shape: {radar_points_new.shape}')
# print('radar_points_new', radar_points_new)

# Project the radar points
projected_points_app = project_points(radar_points_new, K_approximated)
projected_points_cal = project_points(radar_points_new, K_calibrated)
print(f'projected_points_cal shape: {projected_points_cal.shape}')
# print('projected_points_cal', projected_points_cal)

c = projected_points_cal[:, 2]

plt.xlim(0, img_w)
plt.ylim(img_h, 0)  # Invert Y axis to match image coordinate system
# plt.scatter(projected_points_app[:, 0], projected_points_app[:, 1], c='r', label='Radar Projections Approximated')
plt.scatter(projected_points_cal[:, 0], projected_points_cal[:, 1], c=c, cmap='viridis', label='Radar Projections Calibrated')
plt.imshow(img)

plt.title("Projected Radar Points on Image Plane")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.grid(True)
plt.legend()
plt.show()