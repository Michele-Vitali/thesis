import runpy
from pathlib import Path

from register_prob3 import register_prob3


# ================================================================
# REGISTER CUSTOM P-Rob3 DATASET
# ================================================================

register_prob3()


# ================================================================
# PATHS
# ================================================================

THIS_FILE = Path(__file__).resolve()

THESIS_ROOT = THIS_FILE.parents[2]

OPENVLA_FINETUNE = (
    THESIS_ROOT
    / "tools"
    / "openvla"
    / "vla-scripts"
    / "finetune.py"
)


# ================================================================
# RUN OFFICIAL OPENVLA FINETUNER
#
# sys.argv is intentionally preserved.
# Therefore all command-line arguments passed to this wrapper are
# parsed directly by OpenVLA / draccus.
# ================================================================

def main():
    if not OPENVLA_FINETUNE.exists():
        raise FileNotFoundError(
            f"OpenVLA finetune script not found: "
            f"{OPENVLA_FINETUNE}"
        )

    runpy.run_path(
        str(
            OPENVLA_FINETUNE
        ),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
