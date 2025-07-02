
import subprocess
import time

start_time = time.time()

# > ./mmWaveStudio.exe /lua "G:/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/startup_capture.lua"
# subprocess.run(
#     [
#         r"C:/ti/mmwave_studio_03_01_04_04/mmWaveStudio/RunTime/mmWaveStudio.exe",
#         "-lua",
#         r"./manual_init.lua"
#     ]
# )

directory = "C:/ti/mmwave_studio_03_01_04_04/mmWaveStudio/RunTime"
command = ["C:/ti/mmwave_studio_03_01_04_04/mmWaveStudio/RunTime/mmWaveStudio.exe", "/lua", "G:/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/startup_capture.lua"] #Example command

subprocess.run(command, cwd=directory)


end_time = time.time()

print(f'start_time: {start_time}, end_time: {end_time}')