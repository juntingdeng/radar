import json

with open('config.json', 'r') as f:
    cfg = json.load(f)

with open(cfg["msgfile"], 'w') as f:
    f.write("stop")