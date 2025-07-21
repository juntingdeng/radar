from utils.capon import *

LIGHT_SPEED = 3e8
class radarConfig:
    def __init__(self):
        self.num_tx = 1
        self.num_rx = 4
        self.start_freq = 77 # GHz
        self.idle_time = 10 # usec
        self.adc_start_time = 6 # usec
        self.ramp_end_time = 120 # usec, ramp_end_time = adc_start_time + adc_sample_time(per chirp) + extral_time
        self.chirp_slope =  29.982 # MHz/usec
        self.num_adc_samples = 128
        self.adc_sample_rate = 10 # Msps
        self.num_frames = 256
        self.num_chirps = 255 # per TX
        self.start_chirp = 0
        self.end_chirp = 0
        self.data_port = 4098
        self.cfg_port = 4096
        self.SysSrcIP = "192.168.33.30"
        self.FpgaDextIP = "192.168.33.180"
        self.msgfile = "G:\\My Drive\\CMU\\Research\\3DImage\\sensor\\TI\\setup_test\\code\\radar\\test\\msg"
        self.angles = ['Elevation', 'Azimuth']

    def load_cfg(self, cfg_file):
        ## Parse radar configuration from .lua file
        with open(cfg_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.split('(')[0] == 'ar1.ChanNAdcConfig':
                params = line.split('(')[1].split(')')[0]
                params = params.split(',')
                params = [int(p) for p in params]
                self.num_tx = sum(params[: 4])
                self.num_rx = sum(params[4: 8])
            
            elif line.split('(')[0] == 'ar1.ProfileConfig':
                params = line.split('(')[1].split(')')[0]
                params = params.split(',')
                params = [float(p) for p in params]
                self.start_freq = params[1]
                self.idle_time = params[2]
                self.adc_start_time = params[3]
                self.ramp_end_time = params[4]
                self.chirp_slope = params[13]
                self.num_adc_samples = int(params[15])
                # self.num_adc_samples = 1024
                self.adc_sample_rate = params[16]/1000
            
            elif line.split('(')[0] == 'ar1.FrameConfig':
                params = line.split('(')[1].split(')')[0]
                params = params.split(',')
                params = [float(p) for p in params]
                self.start_chirp = int(params[0])
                self.end_chirp = int(params[1])
                self.num_frames = int(params[2])
                self.num_chirps = int(params[3])
                # self.num_chirps = 128
            
            elif line.split('(')[0] == 'ar1.CaptureCardConfig_EthInit':
                params = line.split('(')[1].split(')')[0]
                params = params.split(',')
                self.SysSrcIP = params[0]
                self.FpgaDextIP = params[1]
                self.cfg_port = int(params[3])
                self.data_port = int(params[4])

        assert self.end_chirp - self.start_chirp + 1 == self.num_tx, 'Number of Tx does not match number of chirps'

    def parse_radar(self, cfg_file='./mmWaveStudio/startup_capture.lua'):
        self.load_cfg(cfg_file=cfg_file)

        self.frame_duration = (self.ramp_end_time+self.idle_time)*self.num_chirps*1e-6 # sec
        self.frame_act_ratio = (self.num_adc_samples/self.adc_sample_rate)/(self.ramp_end_time+self.idle_time)

        self.num_chirps = self.num_chirps*self.num_tx
        self.samples_per_frame = self.num_adc_samples*self.num_chirps*self.num_rx
        self.packets_per_frame = (2*self.samples_per_frame)//(728*2)

        self.bandwidth = self.chirp_slope*self.ramp_end_time*1e6
        self.range_resolution = LIGHT_SPEED/(2*self.bandwidth)
        self.max_range = self.range_resolution * self.num_adc_samples

        self.chirp_interval = (self.ramp_end_time + self.idle_time) * 1e-6 # usec
        self.doppler_resolution = compute_doppler_resolution(self.num_chirps, self.bandwidth, self.chirp_interval, self.num_tx)
        self.max_doppler = self.doppler_resolution * self.num_chirps / 2

        self.ranges_real = np.arange(0, self.max_range/2 + self.range_resolution, self.range_resolution)
        range_ticks_real = np.arange(0, len(self.ranges_real), len(self.ranges_real)//10)
        range_tick_labels_real = self.ranges_real[::len(self.ranges_real)//10].round(2)[::-1]
        
        self.num_ant = [self.num_tx, self.num_rx*self.num_tx]
        self.steering_vectors =[compute_steering_vector(num_ant=n, angle_res=1.0, angle_rng=90) for n in self.num_ant]
        attrs = ','.join([f'({key}: {val:.2f})' for key, val in vars(self).items() if isinstance(val, (int, float))])
        print(attrs)
        
        return range_ticks_real, range_tick_labels_real
    
    