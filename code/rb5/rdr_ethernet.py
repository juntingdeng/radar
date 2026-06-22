#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import struct
import sys
import time

RUN = True


def handle_sigint(signum, frame):
    global RUN
    RUN = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture DCA1000 UDP data on Linux/RB5"
    )
    p.add_argument("--bind-ip", default="0.0.0.0",
                   help="IP address to bind to, e.g. 192.168.33.30")
    p.add_argument("--port", type=int, default=4098,
                   help="UDP port for DCA1000 data (default: 4098)")
    p.add_argument("--out", default="data.bin",
                   help="Output file path")
    p.add_argument("--meta-log", default="capture_meta.txt",
                   help="Metadata log path")
    p.add_argument("--bufsize", type=int, default=65535,
                   help="UDP receive buffer size")
    p.add_argument("--timeout", type=float, default=1.0,
                   help="Socket timeout in seconds")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Stop after this many seconds; 0 = until Ctrl+C")
    p.add_argument("--strip-header", action="store_true",
                   help="Strip 10-byte DCA1000 UDP header before writing")
    return p.parse_args()


def main():
    global RUN
    args = parse_args()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.settimeout(args.timeout)
    sock.bind((args.bind_ip, args.port))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_log) or ".", exist_ok=True)

    start_time = time.time()
    pkt_count = 0
    byte_count = 0
    dropped_packets = 0
    first_seq = None
    last_seq = None

    print(f"Listening on {args.bind_ip}:{args.port}")
    print(f"Writing data to: {args.out}")
    print(f"Writing metadata to: {args.meta_log}")
    print("Press Ctrl+C to stop.")

    with open(args.out, "wb") as fout, open(args.meta_log, "w") as flog:
        flog.write("timestamp_sec,src_ip,src_port,seq_num,byte_count_field,payload_len,missing_packets\n")

        while RUN:
            if args.seconds > 0 and (time.time() - start_time) >= args.seconds:
                break

            try:
                packet, addr = sock.recvfrom(args.bufsize)
            except socket.timeout:
                continue

            pkt_count += 1

            # DCA1000 UDP packet header is commonly treated as:
            # 4-byte sequence number + 6-byte byte count
            # Save a log even if packet is shorter than expected.
            seq_num = None
            byte_count_field = None
            missing = 0

            if len(packet) >= 10:
                seq_num = struct.unpack("<I", packet[:4])[0]
                byte_count_field = int.from_bytes(packet[4:10], byteorder="little", signed=False)

                if first_seq is None:
                    first_seq = seq_num
                if last_seq is not None and seq_num > last_seq + 1:
                    missing = seq_num - last_seq - 1
                    dropped_packets += missing
                last_seq = seq_num

                payload = packet[10:] if args.strip_header else packet
            else:
                payload = packet

            fout.write(payload)
            byte_count += len(payload)

            flog.write(
                f"{time.time():.6f},{addr[0]},{addr[1]},"
                f"{'' if seq_num is None else seq_num},"
                f"{'' if byte_count_field is None else byte_count_field},"
                f"{len(payload)},{missing}\n"
            )

            if pkt_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate_mbps = (byte_count * 8 / 1e6) / elapsed if elapsed > 0 else 0.0
                print(
                    f"Packets: {pkt_count}, Bytes written: {byte_count}, "
                    f"Dropped(est): {dropped_packets}, Rate: {rate_mbps:.2f} Mb/s"
                )

    elapsed = time.time() - start_time
    rate_mbps = (byte_count * 8 / 1e6) / elapsed if elapsed > 0 else 0.0

    print("\nCapture finished.")
    print(f"Elapsed time: {elapsed:.2f} s")
    print(f"Packets received: {pkt_count}")
    print(f"Bytes written: {byte_count}")
    print(f"Estimated dropped packets: {dropped_packets}")
    print(f"Average rate: {rate_mbps:.2f} Mb/s")
    if first_seq is not None and last_seq is not None:
        print(f"Sequence range: {first_seq} -> {last_seq}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass