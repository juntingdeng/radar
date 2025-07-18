import numpy as np

LIGHT_SPEED = 299792458 # m/s

def forward_backward_avg(Rxx):
    """ Performs forward backward averaging on the given input square matrix
    Args:
        Rxx (ndarray): A 2D-Array square matrix containing the covariance matrix for the given input data
    Returns:
        R_fb (ndarray): The 2D-Array square matrix containing the forward backward averaged covariance matrix
    """
    assert np.size(Rxx, 0) == np.size(Rxx, 1)

    # --> Calculation
    M = np.size(Rxx, 0)  # Find number of antenna elements
    Rxx = np.matrix(Rxx)  # Cast np.ndarray as a np.matrix

    # Create exchange matrix
    J = np.eye(M)  # Generates an identity matrix with row/col size M
    J = np.fliplr(J)  # Flips the identity matrix left right
    J = np.matrix(J)  # Cast np.ndarray as a np.matrix

    R_fb = 0.5 * (Rxx + J * np.conjugate(Rxx) * J)

    return np.array(R_fb)


def aoa_capon(x, a):
    """ Computes Capon Spectrum
        Inputs:
            x - output of 1D range FFT (v_rtx, num_chirps)
            a - steering vector (num_angles, v_rtx)
        Outputs:
            capon_spectrum - Computed Capon Spectrum (num_angles)
    """
    # perturbation
    p = np.eye(x.shape[0]) * 1e-9   # source covariance

    Rxx = x @ np.conj(x).T
    Rxx = forward_backward_avg(Rxx)

    Rxx_inv = np.linalg.inv(Rxx + p)

    capon_spec = np.reciprocal(np.einsum('ij,ij->i', a.conj(), (Rxx_inv @ a.T).T))
    
    return capon_spec


def compute_steering_vector(num_ant=8, angle_res=1, angle_rng=90):
    """ Computes array of Steering Vectors for a desired angluar range
        and resolution. **This is a special function that only computes the
        steering vectors along a 1D linear axis.**
        Inputs:
            angle_res - angle resolution in degrees
            angle_rng - single sided angle range
            num_ant - number of virtual antennas
        Output:
            steering_vectors
    """
    # get number of steering vectors based on desired angle range and resolution
    num_vec = (2 * angle_rng / angle_res + 1)
    num_vec = int(round(num_vec))

    # convert to radians
    angle_rng = angle_rng*np.pi/180
    angle_res = angle_res*np.pi/180

    # compute steering vectors
    steering_vectors = np.zeros((num_vec, num_ant), dtype=np.complex64)
    for k in range(num_vec):
        for m in range(num_ant):
            steering_vectors[k, m] = np.exp(-1j*np.pi*m
                                            *np.sin(-angle_rng + k*angle_res))
            
    return steering_vectors

def compute_steering_vector_nonuniform(positions=None, angle_ress=1, angle_rngs=[90]):
    """ Computes array of Steering Vectors for a desired angluar range
        and resolution. **This is a special function that only computes the
        steering vectors along a 1D linear axis.**
        Inputs:
            angle_res - angle resolution in degrees
            angle_rng - single sided angle range
            positions - position (x, z) of virtual channels, unit wavelength lambda
        Output:
            steering_vectors
    """
    # get number of steering vectors based on desired angle range and resolution
    angle_rngs = angle_rngs if len(angle_rngs)==2 else angle_rngs+angle_rngs
    angle_ress = angle_ress if len(angle_ress)==2 else angle_ress+angle_ress

    num_vecs = []
    angles = []
    # convert to radians
    for angle_rng, angle_res in zip(angle_rngs, angle_ress):
        num_vecs.append(int(round(2 * angle_rng / angle_res + 1)))
        angle_rng = angle_rng*np.pi/180
        angle_res = angle_res*np.pi/180

        angles.append(np.arange(start=-angle_rng, stop=angle_rng+angle_res, step=angle_res))

    sv_size = (positions.shape[0],) + tuple(angles[i].shape[0] for i in range(len(angles)))
    steering_vectors = np.zeros(sv_size, dtype=np.complex64)

    for phi_i in range(num_vecs[0]): ## azimuth
        phi = angles[0][phi_i]
        for theta_i in range(num_vecs[1]): ## elevation
            theta = angles[1][theta_i]

            u_y = np.cos(theta) * np.sin(phi)
            u_z = np.sin(theta)
            phase = 2 * np.pi * (positions[:, 0] * u_y + positions[:, 1] * u_z)
            steering_vectors[:, phi_i, theta_i] = np.exp(-1j * phase)
            
    return steering_vectors

def aoa_capon_nonuniform(x, a):
    """ Computes Capon Spectrum
        Inputs:
            x - output of 1D range FFT (v_rtx_positions, num_chirps)
            a - steering vector (v_rtx_positions, num_thetas, num_phis)
        Outputs:
            capon_spectrum - Computed Capon Spectrum (num_angles, num_angles)
    """
    # perturbation
    p = np.eye(x.shape[0]) * 1e-9   # source covariance

    Rxx = x @ np.conj(x).T
    Rxx = forward_backward_avg(Rxx)

    Rxx_inv = np.linalg.inv(Rxx + p)

    capon_spec = np.zeros((a.shape[1], a.shape[2]))
    for theta_i in range(a.shape[1]):
        for phi_i in range(a.shape[2]):
            vec = a[:, theta_i, phi_i]
            capon_spec[theta_i, phi_i] = 1/(vec.conj().T @ (Rxx_inv @ vec))
    
    return capon_spec

def compute_range_resolution(adc_sample_rate, num_adc_samples, chirp_slope):
    """
        Compute Range Resolution in meters
        Inputs:
            adc_sample_rate (Msps)
            num_adc_samples (unitless)
            chirp_slope (MHz/usec)
        Outputs:
            range_resolution (meters)
    """

    # compute ADC sample period T_c in msec
    adc_sample_period = (1 / adc_sample_rate) * num_adc_samples # usec
    print(f'adc sample period: {adc_sample_period} usec')

    # next compute the Bandwidth in MHz
    bandwidth = adc_sample_period * chirp_slope # MHz
    print(f'bandwidth:{bandwidth} MHz')

    # Coompute range resolution in meters
    range_resolution = LIGHT_SPEED / (2 * (bandwidth * 1e6)) # meters

    return range_resolution, bandwidth


def compute_doppler_resolution(num_chirps, bandwidth, chirp_interval, num_tx):
    """
        Compute Doppler Resolution in meters/second
        Inputs:
            num_chirps
            bandwidth - bandwidth of each chirp (MHz)
            chirp_interval - total interval of a chirp including idle time (usec)
            num_tx
        Outputs:
            doppler_resolution (ms)
    """
    # compute center frequency in GHz
    center_freq = (77 + (bandwidth*1e-3)/2) # GHz

    # compute center wavelength 
    lmbda = LIGHT_SPEED/(center_freq * 1e9) # meters

    # compute doppler resolution in meters/second
    doppler_resolution = lmbda / (2 * num_chirps * num_tx * chirp_interval)

    return doppler_resolution