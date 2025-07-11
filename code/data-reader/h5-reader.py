import h5py
import os

# === Configurable variables ===
folder_name = 'radar'
h5_base_name = '2025.07.08-17.59.24'  # Without .h5 extension

# === Construct the path to the .h5 file ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # data-reader/
h5_path = os.path.join(base_dir, '..', folder_name, f"{h5_base_name}.h5")

# === Load and inspect the .h5 file ===
def print_h5_contents(h5_file, indent=0):
    for key in h5_file:
        item = h5_file[key]
        if isinstance(item, h5py.Dataset):
            print("  " * indent + f"{key}: shape = {item.shape}, dtype = {item.dtype}")
        elif isinstance(item, h5py.Group):
            print("  " * indent + f"{key}/ (group)")
            print_h5_contents(item, indent + 1)

with h5py.File(h5_path, 'r') as f:
    print(f"Contents of {h5_base_name}.h5 from {folder_name}:")
    print_h5_contents(f)
