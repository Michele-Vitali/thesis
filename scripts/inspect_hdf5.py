import sys

import h5py


def print_group(group, indent=0):
    
    prefix = ("    " * indent)

    for key in group.keys():
        value = group[key]

        if isinstance(value, h5py.Dataset):
            print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{prefix}{key}/")
            print_group(value, indent + 1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_hdf5.py <path/demo.hdf5>")

        return

    path = sys.argv[1]

    with h5py.File(path, "r") as file:
        print(f"File: {path}")
        print_group(file)

if __name__ == "__main__":
    main()