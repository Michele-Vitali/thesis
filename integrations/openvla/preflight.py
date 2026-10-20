import os
from pathlib import Path

# ================================================================
# ENVIRONMENT
# ================================================================

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)


# ================================================================
# PATHS
# ================================================================

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

DATA_ROOT = (
    THESIS_ROOT
    / "rlds_data"
    / "rlds"
)

PREFLIGHT_ROOT = (
    THESIS_ROOT
    / "preflight"
    / "openvla"
)

PREFLIGHT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_ID = "openvla/openvla-7b"
DATASET_NAME = "prob3_pick_place"

LORA_RANK = 32


# ================================================================
# REGISTER P-ROB3 DATASET
# ================================================================

from register import register_prob3
from data_policy import install_prob3_train_only_statistics

register_prob3()
install_prob3_train_only_statistics()


# ================================================================
# IMPORTS
# ================================================================

import numpy as np
import torch

from peft import (LoraConfig, get_peft_model)

from torch.utils.data import DataLoader

from transformers import (AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor)

from prismatic.extern.hf.configuration_prismatic import (OpenVLAConfig)

from prismatic.extern.hf.modeling_prismatic import (OpenVLAForActionPrediction)

from prismatic.extern.hf.processing_prismatic import (PrismaticImageProcessor, PrismaticProcessor)

from prismatic.models.backbones.llm.prompting import (PurePromptBuilder)

from prismatic.util.data_utils import (PaddedCollatorForActionPrediction)

from prismatic.vla.action_tokenizer import (ActionTokenizer)

from prismatic.vla.datasets import (RLDSBatchTransform, RLDSDataset)

from prismatic.vla.datasets.rlds.utils.data_utils import (save_dataset_statistics)


# ================================================================
# HELPERS
# ================================================================

def print_separator(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():
    print_separator("P-Rob3 / OpenVLA-7B PRE-FLIGHT")

    print("PyTorch:", torch.__version__)

    print("CUDA runtime:", torch.version.cuda)

    print("CUDA available:", torch.cuda.is_available())

    print("GPU count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

    device = torch.device("cuda:0")

    # ============================================================
    # REGISTER OPENVLA HF CLASSES
    # ============================================================

    AutoConfig.register("openvla", OpenVLAConfig)

    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # ============================================================
    # LOAD PROCESSOR
    # ============================================================

    print_separator("LOADING PROCESSOR")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Processor loaded: OK")

    # ============================================================
    # LOAD OPENVLA-7B
    # ============================================================

    print_separator("LOADING OPENVLA-7B")

    vla = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

    print("Model loaded on CPU: OK")

    print("Image sizes:", vla.config.image_sizes)

    print("Moving model to:", device)

    vla = vla.to(device)

    print("Model on GPU: OK")

    # ============================================================
    # APPLY LoRA
    #
    # Same configuration used by official finetune.py.
    #
    # NO TRAINING happens here.
    # ============================================================

    print_separator("ATTACHING LoRA")

    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=min(LORA_RANK, 16), lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian")

    vla = get_peft_model(vla, lora_config)

    vla.print_trainable_parameters()

    trainable_parameters = sum(p.numel() for p in vla.parameters() if p.requires_grad)

    total_parameters = sum(p.numel() for p in vla.parameters())

    print("Trainable parameters:", f"{trainable_parameters:,}")

    print("Total parameters:", f"{total_parameters:,}")

    print("Trainable percentage:", (100.0 * trainable_parameters / total_parameters))

    # ============================================================
    # ACTION TOKENIZER
    # ============================================================

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ============================================================
    # RLDS -> OPENVLA TRANSFORM
    # ============================================================

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=(
            processor
            .image_processor
            .apply_transform
        ),
        prompt_builder_fn=PurePromptBuilder,
    )

    # ============================================================
    # LOAD P-ROB3 RLDS DATASET
    # ============================================================

    print_separator(
        "LOADING P-Rob3 RLDS"
    )

    vla_dataset = RLDSDataset(
        DATA_ROOT,
        DATASET_NAME,
        batch_transform,
        resize_resolution=tuple(
            vla.config.image_sizes
        ),

        # Small for current 4-demo smoke-test.
        shuffle_buffer_size=64,

        train=True,
        image_aug=False,
    )

    print(
        "Dataset loaded: OK"
    )

    print(
        "Dataset length:",
        len(vla_dataset),
    )


    # ============================================================
    # VERIFY RLDS SPLITS AND TRAIN-ONLY STATISTICS
    # ============================================================

    import tensorflow_datasets as tfds

    builder = tfds.builder(
        DATASET_NAME,
        data_dir=str(DATA_ROOT),
    )

    required_splits = {"train", "val", "test"}
    actual_splits = set(builder.info.splits.keys())

    if not required_splits.issubset(actual_splits):
        raise RuntimeError(
            f"Missing required RLDS splits: "
            f"{required_splits - actual_splits}"
        )

    expected_train_episodes = int(
        builder.info.splits["train"].num_examples
    )

    statistics_episodes = int(
        vla_dataset.dataset_statistics[DATASET_NAME][
            "num_trajectories"
        ]
    )

    print("RLDS train episodes:", expected_train_episodes)
    print("Statistics trajectories:", statistics_episodes)

    if statistics_episodes != expected_train_episodes:
        raise RuntimeError(
            "Normalization statistics were not computed "
            "from train only! "
            f"Expected {expected_train_episodes} trajectories, "
            f"found {statistics_episodes}."
        )

    print("Train-only normalization: VERIFIED")


    # ============================================================
    # DATASET STATISTICS
    # ============================================================

    print_separator(
        "ACTION NORMALIZATION STATISTICS"
    )

    stats = (
        vla_dataset
        .dataset_statistics[
            DATASET_NAME
        ][
            "action"
        ]
    )

    for key in (
        "mean",
        "std",
        "min",
        "max",
        "q01",
        "q99",
        "mask",
    ):
        if key in stats:
            print(
                f"{key}:",
                np.asarray(
                    stats[key]
                ),
            )

    expected_mask = np.asarray(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]
    )

    actual_mask = np.asarray(
        stats["mask"]
    )

    if not np.array_equal(
        actual_mask,
        expected_mask,
    ):
        raise RuntimeError(
            "Unexpected OpenVLA action "
            f"normalization mask: {actual_mask}"
        )

    print()
    print(
        "Action normalization mask: OK"
    )

    print(
        "First 6 EEF dimensions normalized."
    )

    print(
        "Gripper dimension NOT normalized."
    )

    save_dataset_statistics(
        vla_dataset.dataset_statistics,
        PREFLIGHT_ROOT,
    )

    print(
        "Statistics saved to:",
        PREFLIGHT_ROOT,
    )

    # ============================================================
    # DATALOADER
    # ============================================================

    collator = (
        PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    print_separator(
        "FETCHING ONE REAL P-Rob3 BATCH"
    )

    batch = next(
        iter(
            dataloader
        )
    )

    print(
        "input_ids:",
        batch[
            "input_ids"
        ].shape,
    )

    print(
        "attention_mask:",
        batch[
            "attention_mask"
        ].shape,
    )

    print(
        "pixel_values:",
        batch[
            "pixel_values"
        ].shape,
    )

    print(
        "labels:",
        batch[
            "labels"
        ].shape,
    )

    supervised_tokens = (
        batch["labels"]
        != -100
    ).sum().item()

    print(
        "Supervised action tokens:",
        supervised_tokens,
    )

    if supervised_tokens == 0:
        raise RuntimeError(
            "No supervised action tokens "
            "found in the batch."
        )

    # ============================================================
    # FORWARD PASS ONLY
    #
    # IMPORTANT:
    #   no optimizer
    #   no backward()
    #   no parameter update
    # ============================================================

    print_separator(
        "FORWARD PASS (NO TRAINING)"
    )

    vla.eval()

    with torch.no_grad():

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):

            output = vla(
                input_ids=(
                    batch[
                        "input_ids"
                    ].to(
                        device
                    )
                ),

                attention_mask=(
                    batch[
                        "attention_mask"
                    ].to(
                        device
                    )
                ),

                pixel_values=(
                    batch[
                        "pixel_values"
                    ]
                    .to(
                        torch.bfloat16
                    )
                    .to(
                        device
                    )
                ),

                labels=(
                    batch[
                        "labels"
                    ].to(
                        device
                    )
                ),
            )

    loss = float(
        output.loss
        .detach()
        .cpu()
    )

    print(
        "Forward loss:",
        loss,
    )

    if not np.isfinite(
        loss
    ):
        raise RuntimeError(
            "Forward loss is not finite."
        )

    print_separator(
        "ALL PRE-FLIGHT CHECKS PASSED"
    )

    print(
        "P-Rob3 RLDS        : OK"
    )

    print(
        "OpenVLA transforms : OK"
    )

    print(
        "Action normalization: OK"
    )

    print(
        "OpenVLA-7B        : OK"
    )

    print(
        "LoRA rank 32       : OK"
    )

    print(
        "GPU forward pass   : OK"
    )

    print()
    print(
        "NO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
