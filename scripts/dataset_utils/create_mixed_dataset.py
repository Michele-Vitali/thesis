import h5py
import numpy as np

def create_mixed_hdf5():
    
    clean_hdf5_path = "/home/vlabot/Scrivania/thesis/hdf5_data/anchor/100_clean.hdf5"
    rand_hdf5_path = "path al file hdf5 con 300 random splittato!"
    
    
    output_hdf5_path = "path alla cartella del nuovo dataset!"

    with h5py.File(clean_hdf5_path, 'r') as f_clean, h5py.File(rand_hdf5_path, 'r') as f_rand, h5py.File(output_hdf5_path, 'w') as f_out:

        N_FROM_ANCHOR = 100
        N_FROM_RANDOM_TRAIN = 300
        # 1. Copy the first N trajectories from the anchor.
        clean_keys = list(f_clean['data'].keys())
        for i in range(N_FROM_ANCHOR):
            key = clean_keys[i]
            # Copy the trajectory group while renaming it with 'clean_' in front.
            f_clean.copy(f'data/{key}', f_out, f'data/clean_{key}')
            print(f"Copied anchor clean: {key}")

        # 2. Copy the first N trajectories for training from the randomized dataset.
        rand_keys = list(f_rand['data'].keys())
        for i in range(N_FROM_RANDOM_TRAIN):
            key = rand_keys[i]
            f_rand.copy(f'data/{key}', f_out, f'data/rand_train_{key}')
            print(f"Copied random train: {key}")
            
        # 3. Copia le ultime 50 traiettorie RANDOMIZZATE (Validation)
        for i in range(300, 350):
            key = rand_keys[i]
            f_rand.copy(f'data/{key}', f_out, f'data/rand_val_{key}')
            print(f"Copiata random val: {key}")

    print(f"File ibrido creato in: {output_hdf5_path}")
    print("Contiene: 100 Anchor Clean + 300 Rand Train + 50 Rand Val")

if __name__ == "__main__":
    create_mixed_hdf5()