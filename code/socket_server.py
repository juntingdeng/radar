import socket
import subprocess
import time
# inet 172.26.112.133
s = socket.socket()
# s.bind(("192.168.33.30", 5005))
s.bind(("172.26.37.236", 5005))
s.listen(1)
conn, addr = s.accept()
message = conn.recv(1024).decode()
print("Connected by", addr)
print("Message:", message)

directory = "C:/ti/mmwave_studio_03_01_04_04/mmWaveStudio/RunTime"
command = ["C:/ti/mmwave_studio_03_01_04_04/mmWaveStudio/RunTime/mmWaveStudio.exe", "/lua", "G:/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/startup_capture.lua"] #Example command

if message == "Start":
    start_time = time.time()
    subprocess.run(command, cwd=directory)
    end_time = time.time()

    print(f'start_time: {start_time}, end_time: {end_time}')

conn.close()