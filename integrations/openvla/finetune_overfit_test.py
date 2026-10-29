import argparse
import runpy
import sys
from pathlib import Path

# Register custom policy and P-Rob3 dataset configuration
from data_policy import install_prob3_train_only_statistics
from register import DATASET_NAME, RLDS_DATA_ROOT, register_prob3

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]
OPENVLA_FINETUNE = THESIS_ROOT / "tools" / "openvla" / "vla-scripts" / "finetune.py"


def main():
    if not OPENVLA_FINETUNE.exists():
        raise FileNotFoundError(f"Fine-tune script not found: {OPENVLA_FINETUNE}")

    # Extract custom wrapper arguments before forwarding sys.argv to finetune.py
    filtered_args = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        filtered_args.append(arg)
        i += 1

    # Register prob3_pick_place dataset in OpenVLA
    register_prob3()

    # Install train-only policy
    install_prob3_train_only_statistics()

    # Ensure required default CLI arguments if missing
    if "--dataset_name" not in filtered_args:
        filtered_args.extend(["--dataset_name", DATASET_NAME])

    if "--data_root_dir" not in filtered_args:
        filtered_args.extend(["--data_root_dir", str(RLDS_DATA_ROOT)])

    # Forward parsed arguments to finetune.py
    sys.argv = [str(OPENVLA_FINETUNE)] + filtered_args

    runpy.run_path(str(OPENVLA_FINETUNE), run_name="__main__")


if __name__ == "__main__":
    main()