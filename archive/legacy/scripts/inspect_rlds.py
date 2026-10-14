import tensorflow_datasets as tfds
import numpy as np


DATASET_PATH = ("/home/vlabot/Scrivania/thesis/data/rlds/prob3_pick_place/1.0.0")


def main():
    print("=" * 60)
    print("Loading RLDS dataset...")
    print("=" * 60)

    builder = tfds.builder_from_directory(DATASET_PATH)

    print(builder.info)

    print("\nAvailable splits:")
    print(builder.info.splits)

    # Load train split.
    dataset = builder.as_dataset(split="train")

    episode = next(iter(tfds.as_numpy(dataset)))

    print("\n" + "=" * 60)
    print("EPISODE STRUCTURE")
    print("=" * 60)

    print("Episode keys:", episode.keys())

    steps = list(episode["steps"])

    print("Number of steps:", len(steps))

    if len(steps) == 0:
        raise RuntimeError("Episode contains zero steps.")

    print("Step keys:", steps[0].keys())

    print("Observation keys:", steps[0]["observation"].keys())

    images = np.stack([step["observation"]["image"] for step in steps], axis=0)

    states = np.stack([step["observation"]["state"] for step in steps], axis=0)

    actions = np.stack([step["action"] for step in steps], axis=0)

    rewards = np.asarray([step["reward"] for step in steps], dtype=np.float32)

    discounts = np.asarray([step["discount"] for step in steps], dtype=np.float32)

    is_first = np.asarray([step["is_first"] for step in steps], dtype=bool)

    is_last = np.asarray([step["is_last"] for step in steps], dtype=bool)

    is_terminal = np.asarray([step["is_terminal"] for step in steps], dtype=bool)

    language = [step["language_instruction"] for step in steps]

    print("\n" + "=" * 60)
    print("SHAPES")
    print("=" * 60)

    print("Images:", images.shape, images.dtype)

    print("States:", states.shape, states.dtype)

    print("Actions:", actions.shape, actions.dtype)

    print("Rewards:", rewards.shape)

    print("Discounts:", discounts.shape)

    print("\n" + "=" * 60)
    print("LANGUAGE")
    print("=" * 60)

    first_instruction = language[0]

    if isinstance(first_instruction, bytes):
        first_instruction = first_instruction.decode("utf-8")

    print(first_instruction)

    # Check language consistency.
    unique_instructions = set(instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction) for instruction in language)

    print("Unique instructions:", unique_instructions)

    print("\n" + "=" * 60)
    print("ACTION STATISTICS")
    print("=" * 60)

    np.set_printoptions(precision=4, suppress=True)

    print("Min:", np.min(actions, axis=0))

    print("Max:", np.max(actions, axis=0))

    print("Mean:", np.mean(actions, axis=0))

    print("Std:", np.std(actions, axis=0))

    print("\nFirst 10 actions:")

    print(actions[:10])

    print("\nLast 10 actions:")

    print(actions[-10:])

    print("\n" + "=" * 60)
    print("GRIPPER")
    print("=" * 60)

    unique_gripper = np.unique(actions[:, 6])

    print("Unique gripper values:", unique_gripper)

    print("\n" + "=" * 60)
    print("RLDS FLAGS")
    print("=" * 60)

    print("is_first indices:", np.where(is_first)[0])

    print("is_last indices:", np.where(is_last)[0])

    print("is_terminal indices:", np.where(is_terminal)[0])

    print("Reward == 1 indices:", np.where(rewards > 0.5)[0])

    print("Final discount:", discounts[-1])

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    assert images.ndim == 4
    assert images.shape[1:] == (256, 256, 3)

    assert states.shape == (len(steps), 7)

    assert actions.shape == (len(steps), 7)

    assert np.sum(is_first) == 1

    assert is_first[0]

    assert np.sum(is_last) == 1

    assert is_last[-1]

    assert np.sum(is_terminal) == 1

    assert is_terminal[-1]

    assert set(unique_gripper.tolist()).issubset({0.0, 1.0})

    print("All basic RLDS checks passed!")

    print("=" * 60)

if __name__ == "__main__":
    main()