import time
import numpy as np
import json
import socket
import struct
import tables as tb
from datetime import datetime

PACKET_BUFSIZE = 8192
MAX_PACKET_SIZE = 4096

class Packet(tb.IsDescription):
    """Raw radar packet data type."""
    t = tb.Float64Col()
    packet_data = tb.UInt16Col(shape=(728,))
    packet_num = tb.UInt32Col()
    byte_count = tb.UInt64Col()


class DataCollector:
    """Data collection h5 table management."""

    def __init__(self, h5file):
        scan_group = h5file.create_group('/', 'scan', 'Scan information')
        self.packets = h5file.create_table(
            scan_group, 'packet', Packet, 'Packet data')

        self.total_packets = 0
        self.chunk_packets = 0

        self.start_time = time.time()
    
    def write_packet(self, packet_num, byte_count, packet_data):
        """Write packet to h5 file."""
        self.packets.row['t'] = time.time()
        self.packets.row['packet_data'] = packet_data
        self.packets.row['packet_num'] = packet_num
        self.packets.row['byte_count'] = byte_count
        self.packets.row.append()
        self.chunk_packets += 1

    def flush(self):
        """Flush packets to file."""
        print('[t={:.3f}s] Flushing {} packets.'.format(
            time.time() - self.start_time, self.chunk_packets))
        self.packets.flush()
        self.total_packets += self.chunk_packets
        self.chunk_packets = 0


def _read_data_packet(data_socket):
    """Helper function to read in a single ADC packet via UDP.

    The format is described in the [DCA1000EVM user guide](
        https://www.ti.com/tool/DCA1000EVM#tech-docs)::

        | packet_num (u4) | byte_count (u6) | data ... |

    The packet_num and byte_count appear to be in little-endian order.

    Returns
    -------
    packet_num: current packet number
    byte_count: byte count of data that has already been read
    data: raw ADC data in current packet
    """
    data = data_socket.recv(MAX_PACKET_SIZE)
    # Little-endian, no padding
    packet_num, byte_count = struct.unpack('<LQ', data[:10] + b'\x00\x00')
    packet_data = np.frombuffer(data[10:], dtype=np.int16)
    return packet_num, byte_count, packet_data

def collector():
    global visualize_buffer
    global cfg
    global lock

    _config_socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    _config_socket.bind((cfg.SysSrcIP, cfg.cfg_port))

    data_socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    data_socket.bind((cfg.SysSrcIP, cfg.data_port))
    data_socket.settimeout(10)

    # Instruct the mmWave Studio Lua server to instruct the radar to start
    # collecting data.
    with open(cfg.msgfile, 'w') as f:
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