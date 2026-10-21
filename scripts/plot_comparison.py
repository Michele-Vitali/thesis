
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EVAL_ROOT = Path(__file__).resolve().parents[1] / "evaluations"
RESULT_NAMES = ("evaluation_results.json")


def plot_comparisons(only_models=None, name_contains=None):
    rows = []
    confusion_matrices = {}

    # Load the already computed results.
    for folder in EVAL_ROOT.iterdir():
        # If the element is not a folder or the model is not in the "plotting" ones, just skip the folder...
        if not folder.is_dir():
            continue
        if only_models is not None and folder.name not in only_models:
            continue
        if name_contains and name_contains not in folder.name:
            continue


        # Grab the result's file for that model...
        result_file = folder / RESULT_NAMES
        if result_file is None:
            # If there are no results for that model, just skip it.
            continue

        # Load the json data...
        data = json.loads(result_file.read_text(encoding="utf-8"))

        # If the evaluation data does not come from the validation split, skip the folder!
        if data.get("split") != "val":
            continue

        # Grab the model
        model = data["model"]

        # Grab the model's checkpoint steps...
        match = re.search(r"--(\d+)_chkpt$", model)
        if not match:
            print(f"[SKIP] Cannot identify checkpoint step: {model}")
            continue

        arm = data["metrics"]["arm"]
        grip = data["metrics"]["gripper"]

        # Define the pandas dataframe structure...
        row = {
            "step": int(match.group(1)),
            "run": model.rsplit("--", 1)[0],
            "dataset": data["dataset"],
            "Arm MAE": arm["mae"],
            "Translation MAE": arm["translation_mae"],
            "Rotation MAE": arm["rotation_mae"],
            "Direction cosine": arm["direction_cosine"],
            "Gripper accuracy": grip["accuracy"],
            "Gripper Macro F1": grip["macro_f1"],
            "zero_arm_mae": arm["zero_arm_mae"],
            "zero_translation_mae": arm["zero_translation_mae"],
            "zero_rotation_mae": arm["zero_rotation_mae"],
        }

        for name, value in arm["axis_mae"].items():
            row.update({f"axis_{name}": value})

        row.update({
            "Arm RMSE": arm["rmse"],
            "Translation RMSE": arm["translation_rmse"],
            "Rotation RMSE": arm["rotation_rmse"],
            "Translation vector error": arm["translation_vector_error"],
            "Rotation vector error": arm["rotation_vector_error"],

            # Gripper metrics for each class.
            "Precision closed": grip["precision_closed"],
            "Recall closed": grip["recall_closed"],
            "F1 closed": grip["f1_closed"],
            "Precision open": grip["precision_open"],
            "Recall open": grip["recall_open"],
            "F1 open": grip["f1_open"],

            # Dataset statistics.
            "Total frames": data["metrics"]["num_frames"],
            "Valid direction frames": arm["direction_cosine_valid_frames"],
            "GT closed frames": grip["num_closed_gt"],
            "GT open frames": grip["num_open_gt"],
        })

        confusion_matrices[int(match.group(1))] = grip["confusion_matrix"]

        # Append the model's row to the other rows...
        rows.append(row)

    if len(rows) < 2:
        raise ValueError("At least two checkpoint results are required!")

    # Never compare different runs/datasets accidentally.
    if len({r["run"] for r in rows}) != 1 or len({r["dataset"] for r in rows}) != 1:
        raise ValueError("Selected models belong to different runs or datasets!")

    # Create the pandas dataframe and sort the rows by their checkpoint's steps
    df = pd.DataFrame(rows).sort_values("step")

    # Grab the model's name
    run_note = rows[0]["run"].split("--")[1] if "--" in rows[0]["run"] else rows[0]["run"]

    # Define where the results will be put...
    output_dir = EVAL_ROOT / "comparisons" / run_note
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seaborn settings...
    sns.set_theme(style="whitegrid")

    # Graph 1: Main evaluation metrics.
    specs = [
        ("Arm MAE", "zero_arm_mae"),
        ("Translation MAE", "zero_translation_mae"),
        ("Rotation MAE", "zero_rotation_mae"),
        ("Direction cosine", None),
        ("Gripper accuracy", None),
        ("Gripper Macro F1", None),
    ]

    # Create the layout for the plots...
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Loop through the metrics
    for ax, (metric, baseline) in zip(axes.flat, specs):
        sns.lineplot(data=df, x="step", y=metric, marker="o", ax=ax, label="OpenVLA")

        # If we have a baseline (=> we are looking at arm properties) set it to a gray color...
        if baseline:
            ax.axhline(df[baseline].iloc[0], linestyle="--", color="gray", label="Zero-action baseline")

        # If we do not have a baseline, check if we are considering gripper metrics or the cosine similarity...
        elif metric.startswith("Gripper"):
            ax.set_ylim(0, 1)
        elif metric == "Direction cosine":
            ax.set_ylim(-1, 1)

        # Other graph's settings...
        ax.set_xticks(df["step"])
        ax.set_xlabel("Training steps")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=200)
    plt.close(fig)

    # Graph 2: MAE of each individual axis.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, axis_names, title in zip(axes, [("x", "y", "z"), ("rx", "ry", "rz")], ["Translation axis MAE", "Rotation axis MAE"]):
        # Set the graph's columns
        columns = [f"axis_{name}" for name in axis_names]

        long_df = df[["step", *columns]].melt("step", var_name="axis", value_name="MAE")
        long_df["axis"] = long_df["axis"].str.replace("axis_", "", regex=False)

        sns.lineplot(data=long_df, x="step", y="MAE", hue="axis", marker="o", ax=ax)
        ax.set_xticks(df["step"])
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_dir / "axes.png", dpi=200)
    plt.close(fig)


    def save_metric_grid(metric_names, filename, probabilities=False):
        ncols = 3
        nrows = (len(metric_names) + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(15, 4 * nrows),
            squeeze=False,
        )

        for ax, metric in zip(axes.flat, metric_names):
            sns.lineplot(
                data=df,
                x="step",
                y=metric,
                marker="o",
                ax=ax,
            )

            ax.set_title(metric)
            ax.set_xlabel("Training steps")
            ax.set_ylabel(metric)
            ax.set_xticks(df["step"])

            if probabilities:
                lower = max(0, float(df[metric].min()) - 0.03)
                ax.set_ylim(lower, 1.005)

        for ax in list(axes.flat)[len(metric_names):]:
            ax.set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    save_metric_grid(
        [
            "Arm RMSE",
            "Translation RMSE",
            "Rotation RMSE",
            "Translation vector error",
            "Rotation vector error",
        ],
        "offline_arm_details.png",
    )

    save_metric_grid(
        [
            "Precision closed",
            "Recall closed",
            "F1 closed",
            "Precision open",
            "Recall open",
            "F1 open",
        ],
        "offline_gripper_details.png",
        probabilities=True,
    )


    # Plot the gripper confusion matrix for every checkpoint.
    fig, axes = plt.subplots(
        1,
        len(df),
        figsize=(4.5 * len(df), 4.5),
        squeeze=False,
    )

    max_count = max(
        max(row)
        for matrix in confusion_matrices.values()
        for row in matrix
    )

    for ax, step in zip(axes.flat, df["step"]):
        cm = confusion_matrices[int(step)]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=max_count,
            xticklabels=["Closed", "Open"],
            yticklabels=["Closed", "Open"],
            ax=ax,
        )

        ax.set_title(f"Checkpoint {step:,}")
        ax.set_xlabel("Predicted gripper")
        ax.set_ylabel("Ground truth")

    fig.tight_layout()
    fig.savefig(output_dir / "offline_confusion_matrices.png", dpi=200)
    plt.close(fig)

    # Save a complete numeric comparison table.
    df.to_csv(output_dir / "all_metrics.csv", index=False)


    print(f"[PLOTS] Saved comparisons to: {output_dir}")


if __name__ == "__main__":
    # The contains args is useful for plotting specific models...
    parser = argparse.ArgumentParser()
    parser.add_argument("--contains", default=None)
    args = parser.parse_args()

    plot_comparisons(name_contains=args.contains)
