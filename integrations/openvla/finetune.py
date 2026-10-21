import runpy
from pathlib import Path

from data_policy import install_prob3_train_only_statistics
from register import register_prob3

# Register the custom dataset
register_prob3()
install_prob3_train_only_statistics()   # Ensure the statistics used in training only come from the training split of the dataset!

# Useful paths...
THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]
OPENVLA_FINETUNE = (THESIS_ROOT / "tools" / "openvla" / "vla-scripts" / "finetune.py")

# Run the official finetune.py file but with our modifications included!
def main():
    # Check whether the finetune.py file actually exists...
    if not OPENVLA_FINETUNE.exists():
        raise FileNotFoundError(f"OpenVLA finetune script not found: {OPENVLA_FINETUNE}")

    # Run the script!
    runpy.run_path(str(OPENVLA_FINETUNE), run_name="__main__")

if __name__ == "__main__":
    main()
