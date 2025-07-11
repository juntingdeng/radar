import socket
import subprocess

s = socket.socket()
s.connect(("192.168.33.30", 5005))
#s.sendall(b"Start")

video = "cap4.mp4"

# cam_start = f"gst-launch-1.0 -e qtiqmmfsrc name=camsrc camera=0 do-timestamp=true ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! clockoverlay halignment=left valignment=top shaded-background=false font-desc='Sans, 10' ! queue ! qtic2venc ! queue ! h264parse ! mp4mux ! queue ! filesink location=/home/juntingd/videos/{video}"

cam_start = [
    "gst-launch-1.0", "-e",
    "qtiqmmfsrc", "name=camsrc", "camera=0", "do-timestamp=true",
    "!", "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1",
    "!", "clockoverlay", "halignment=left", "valignment=top",
    "shaded-background=false", "font-desc=Sans, 10",
    "!", "queue",
    "!", "qtic2venc",
    "!", "queue",
    "!", "h264parse",
    "!", "mp4mux",
    "!", "queue",
    "!", f"filesink", f"location=/home/juntingd/videos/{video}"
]

s.sendall(b"Start")
subprocess.run([cam_start])

s.close()
