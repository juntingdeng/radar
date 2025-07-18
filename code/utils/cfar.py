import numpy as np
from scipy.ndimage import convolve1d

def cfar(guard_len, train_len, p_fa, range_azimuth):

    window = (2 * (train_len + guard_len) + 1)
    a = train_len*(p_fa**(-1/train_len) - 1)
    print(f"Threshold scale factor: {a:.4f}")

    cfar_kernel = np.ones(window, dtype=float) / (2*train_len)
    cfar_kernel[train_len: train_len + (2*guard_len) + 1] = 0.

    xall = range_azimuth
    detected = np.ones_like(xall)
    range_bins, angle_bins = xall.shape
    range_dect, angle_dect = [], []
    for bi in range(0, range_bins):
        x = xall[bi]
        noise_level = convolve1d(x, cfar_kernel, mode='nearest')
        threshold = noise_level + np.log10(a)
        d = xall[bi, :] > threshold
        detected[bi] *= d
        range_dect.append(sum(d))
        # print(f'range bin: {bi}, ndetected: {np.sum(d)}')

    for bi in range(0, angle_bins):
        x = xall[:, bi]
        noise_level = convolve1d(x, cfar_kernel, mode='nearest')
        threshold = noise_level + np.log10(a)
        d = xall[:, bi] > threshold
        detected[:, bi] *= d
        angle_dect.append(sum(d))
        # print(f'angle bin: {bi}, ndetected: {np.sum(d)}')
    print(f'range_dect:{sum(range_dect)}, angle_dect: {sum(angle_dect)}')
    print(f'nbins in total:{range_bins*angle_bins}, ndetected: {np.sum(detected)}')
    
    return detected, range_dect, angle_dect

import numpy as np
from scipy.ndimage import uniform_filter

def cfar_3d(power_cube, guard_len=(), train_len=(), p_fa=1e-7, threshold_scale=1.5):

    detected = np.ones_like(power_cube)
    range_bins, azi_bins, ele_bins = power_cube.shape

    cfar_kernels = []
    tha = []
    for i in range(3):
        window = (2 * (train_len[i] + guard_len[i]) + 1)
        tha.append(train_len[i]*(p_fa**(-1/train_len[i]) - 1))
        print(f"Threshold scale factor: {tha[-1]:.4f}")

        cfar_kernel = np.ones(window, dtype=float) / (2*train_len[i])
        cfar_kernel[train_len[i]: train_len[i] + (2*guard_len[i]) + 1] = 0.
        cfar_kernels.append(cfar_kernel)

    range_dect, azi_dect, ele_dect = [], [], []
    for azi in range(azi_bins):
        for ele in range(ele_bins):
            x = power_cube[:, azi, ele]
            noise_level = convolve1d(x, cfar_kernels[0], mode='nearest')
            threshold = noise_level + np.log10(tha[0])
            d = x > threshold
            detected[:, azi, ele] *= d
            range_dect.append(sum(d))
    
    for rgi in range(range_bins):
        for ele in range(ele_bins):
            x = power_cube[rgi, :, ele]
            noise_level = convolve1d(x, cfar_kernels[1], mode='nearest')
            threshold = noise_level + np.log10(tha[1])
            d = x > threshold
            detected[rgi, :, ele] *= d
            azi_dect.append(sum(d))
    
    for rgi in range(range_bins):
        for azi in range(ele_bins):
            x = power_cube[rgi, azi, :]
            noise_level = convolve1d(x, cfar_kernels[2], mode='nearest')
            threshold = noise_level + np.log10(tha[2])
            d = x > threshold
            detected[rgi, azi, :] *= d
            ele_dect.append(sum(d))

    print(f'range_dect:{sum(range_dect)}, azi_dect: {sum(azi_dect)}, ele_dect: {sum(ele_dect)}')
    print(f'nbins in total:{range_bins*azi_bins*ele_bins}, ndetected: {np.sum(detected)}')
    
    return detected



def cfar_3d_cube(power_cube, guard_len=(), train_len=(), p_fa=1e-7, threshold_scale=1.5):
    # Total window size = guard + training on each side
    guard = guard_len
    training = train_len
    
    kernel_size = tuple(2*(g + t) + 1 for g, t in zip(guard, training))
    guard_size = tuple(2*g + 1 for g in guard)

    total_cells = np.prod(kernel_size)
    num_guard_cells = np.prod(guard_size)
    num_training_cells = total_cells - num_guard_cells
    threshold_scale = num_training_cells*(p_fa**(-1/num_training_cells) - 1)
    print(f"Threshold scale factor: {threshold_scale:.4f}")
    
    mean_cube = uniform_filter(power_cube, size=kernel_size, mode='nearest')
    guard_cube = uniform_filter(power_cube, size=guard_size, mode='nearest')

    noise_est = (mean_cube * total_cells - guard_cube * num_guard_cells) / num_training_cells

    # Threshold
    threshold = noise_est + np.log10(threshold_scale)
    detections = (power_cube > threshold)

    return detections.astype(np.uint8)

def radar_points(detecteds, angles=['Azimuth', 'Elevation'], radar=None):
    assert len(detecteds) == len(angles)

    angle_peaks = {k:[] for k in angles}
    angle_peaks = {k:[] for k in angles}
    range_peaks = {k:[] for k in angles}

    for detected, angle in zip(detecteds, angles):

        for y in range(detected.shape[0]):
            for x in range(detected.shape[1]):
                if detected[y][x] == 1:
                    xdis = np.linspace(-90, 90, 181).round(1)[x], 
                    ydis = np.arange(start=0, stop=radar.range_resolution*radar.num_adc_samples//2, step=radar.range_resolution)[::-1][y]
                    angle_peaks[angle].append(xdis)
                    range_peaks[angle].append(ydis)
    
    peak_ranges = range_peaks['Azimuth']
    peak_azimuths = angle_peaks['Azimuth']
    peak_elevations = angle_peaks['Elevation'] if 'Elevation' in angles else 0

    radar_points = []
    for azim, elev, rang in zip(peak_azimuths, peak_elevations, peak_ranges):
        azim = np.pi*azim/180
        elev = np.pi*elev/180
        # AoA estimation
        # Range to Cartesian conversion
        x = rang * np.cos(elev) * np.cos(azim)
        y = rang * np.cos(elev) * np.sin(azim)
        z = rang * np.sin(elev)

        radar_points.append([y, -z, x])

    return radar_points

def radar_points_3d(detecteds, angles_grid, radar=None):
    # detecteds: [ranges, azimuths, elevations]
    # angles_grid: [azimuths, elevations]
    # return: list of [x, y, z] points
    radar_points = []

    ranges = np.arange(start=0, stop=radar.range_resolution*radar.num_adc_samples//2, step=radar.range_resolution)
    for ri in range(detecteds.shape[0]):
        rang = ranges[ri]
        for ai in range(detecteds.shape[1]):
            azim = angles_grid[0][ai]
            for ei in range(detecteds.shape[2]):
                elev = angles_grid[1][ei]
                if detecteds[ri][ai][ei] == True:

                    x = rang * np.cos(elev) * np.cos(azim)
                    y = rang * np.cos(elev) * np.sin(azim)
                    z = rang * np.sin(elev)
                    radar_points.append([y, -z, x])

    return radar_points


# Approximate intrinsic matrix for IMX577 (1280x720)
K_approximated = np.array([
    [1380,    0, 960],
    [   0, 1380, 540],
    [   0,    0,   1]
])

# Calibrated
K_calibrated = np.array([[664.56415003,   0.,        961.50698487],
 [  0.,         667.63344917, 533.49934159],
 [  0.,           0.,         1.        ]])

def project_points(points_3d, K):
    """
    Projects 3D points (in camera coordinates) to 2D image coordinates.
    """
    # Only keep points with Z > 0 (in front of the camera)
    valid = points_3d[:, 2] > 0
    points_3d = points_3d[valid]
    # Convert to homogeneous coordinates
    points_3d_h = points_3d.T  # Shape: (3, N)

    # Project into image
    points_2d_h = K @ points_3d_h  # Shape: (3, N)
    # Normalize to get pixel coordinates
    points_2d = (points_2d_h[:2, :] / points_2d_h[2, :]).T  # Shape: (N, 2)
    range = points_2d_h[2, :][None, ...]
    print(range.shape)
    return np.concatenate([points_2d, range.T], axis=-1)