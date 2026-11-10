import tensorflow as tf
import tensorflow_datasets as tfds
import rlds

# 1. Load the dataset (returns a tf.data.Dataset of episodes)
DATASET_NAME = 'prob3_pick_place'
DATASET_DIR = "/home/vlabot/Scrivania/thesis/rlds_data/rlds_base"  
dataset = tfds.load(name=DATASET_NAME, data_dir=DATASET_DIR, split='train[:10]', download=False) # Loading first 10 episodes

# 2. Iterate through episodes and steps
for episode in dataset:
    # Access episode-level metadata
    episode_id = episode[rlds.EPISODE_ID].numpy()
    print(f"Processing Episode: {episode_id}")
    
    # Access the steps dataset inside the episode
    steps_dataset = episode[rlds.STEPS]
    
    for step in steps_dataset:
        # Each step contains observations, actions, rewards, and discount factors
        observation = step[rlds.OBSERVATION]
        action = step[rlds.ACTION]
        reward = step[rlds.REWARD].numpy()
        is_terminal = step[rlds.IS_TERMINAL].numpy()

        print(observation)
        
    break
