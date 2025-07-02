import numpy as np
import matplotlib.pyplot as plt

def read(path, num_rx=4, num_samples=256, num_chrips=64):
    adc_raw = np.fromfile(path, dtype=np.int16)
    print(f'adc_raw shape:{adc_raw.shape}')

    # adc_raw4 = adc_raw.reshape(4, adc_raw.shape[0]//4)
    # adc_real = adc_raw4[:2]
    # adc_imag = adc_raw4[2:]


    adc_complex = adc_raw#[0::2] + 1j * adc_raw[1::2]
    # print(f'adc_complex shape:{adc_complex.shape}')
    
    # plt.plot(np.real(adc_complex[0]))
    # plt.plot(np.imag(adc_complex[0]))
    # plt.show()

    num_chrips_total = adc_complex.size // (num_rx * num_samples)
    num_frames = num_chrips_total // num_chrips

    adc_data = adc_complex.reshape(num_frames, num_chrips, num_rx, num_samples)

    return adc_data

adc = read('C:\\ti\\mmwave_studio_03_01_04_04\\mmWaveStudio\\PostProc\\adc_data_Raw_1.bin')
plt.plot(adc[0,0,0])

range_profile = np.abs(np.fft.fft(adc[0], axis=-1))
range_profile = range_profile.mean(axis=0).mean(axis=0)

# plt.plot(range_profile)
plt.show()