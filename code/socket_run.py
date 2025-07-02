import socket
import subprocess
import time

login = ["ssh", f"juntingd@qrb5165-rb5", "python3", "socket_client.py"]
cd = ["cd", "/home/juntingd/"]
run = ["python3", "socket_client.py"]
           

subprocess.run(login)
# subprocess.run(cd)
# subprocess.run(["pwd"])
# subprocess.run(["su"])
# subprocess.run(run)