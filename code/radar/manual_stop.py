import json

with open('./radar/config.json', 'r') as f:
    cfg = json.load(f)

with open(cfg["msgfile"], 'w') as f:
    f.write("stop")