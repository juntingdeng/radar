import h5py
import collections
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import threading
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\..\\..\\"))
from utils.parse_config import *
from utils.collection import *
from utils.visualizer import *

def main(args):
    radar = radarConfig()
    range_ticks_real, range_tick_labels_real = radar.parse_radar(cfg_file=args.cfg_file)

    # if Live: data capturing while processing, 
    # else: read captured data from .h5 file
    live = args.live
    visualize_buffer = None
    adc_frames = None
    npeaks = args.npeaks
    lock = None
    if live:
        lock = threading.Lock()
        obj = collector(visualize_buffer=visualize_buffer, radar=radar, lock=lock)
        thread = threading.Thread(target=obj.collect, daemon=True)
        thread.start()
    else:
        obj = None
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
        print(f'adc_raw.shape[0]: {adc_raw.shape[0]}, radar.num_adc_samples: {radar.num_adc_samples}, radar.num_chirps: {radar.num_chirps}, radar.num_rx: {radar.num_rx}, nframes: {nframes}')
        
        # Ensure adc_raw is a multiple of the number of samples
        if adc_raw.shape[0] != adc_raw.shape[0] == radar.num_adc_samples*radar.num_chirps*radar.num_rx*radar.num_frames:
            print('Inconsistent adc_raw shape, adjusting...')
            adc_raw = adc_raw[:radar.num_adc_samples*radar.num_chirps*radar.num_rx*radar.num_frames]

        # assert adc_raw.shape[0] == radar.num_adc_samples*radar.num_chirps*radar.num_rx*radar.num_frames or 0 < nframes <radar.num_frames
        print(f'{nframes} received')
        adc_raw = adc_raw[ :radar.num_adc_samples*radar.num_chirps*radar.num_rx*nframes]
        adc_frames = adc_raw.reshape(nframes, radar.num_chirps, radar.num_rx, radar.num_adc_samples)
        adc_tx = [adc_frames[:, i::radar.num_tx, :, :] for i in range(radar.num_tx)]
        adc_frames = np.concatenate(adc_tx, axis=-2)
        print(f'Caputred data shape:{adc_frames.shape}')

    app = QtWidgets.QApplication([])
    win = MyApp(radar=radar,
                collector=obj,
                live=live,
                lock = lock,
                adc_frames=adc_frames,
                npeaks=npeaks,
                range_ticks_real=range_ticks_real,
                range_tick_labels_real=range_tick_labels_real)

    win.resize(800, 500)
    win.show()
    app.exec_()

def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--cfg_file', default='./code/mmWaveStudio/server.lua')
    args.add_argument('--data_file', default=None, type=str)
    args.add_argument('--live', action='store_true')
    args.add_argument('--cfar', action='store_true')
    args.add_argument('--npeaks', default=16, type=int)
    
    return args.parse_args()

if __name__ == '__main__':
    args = args_parser()
    main(args)