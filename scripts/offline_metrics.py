import numpy as np

ARM_AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")

def classification_metrics(tp: int, fp: int, fn: int) -> dict:
    """Computes precision, recall and F1 for a binary class.
    
    Args:
        tp (int): The number of the True Positive occurences.
        fp (int): The number of the False Positive occurences.
        fn (int): The number of the False Negatives occurences.
    
    Returns:
        dict: A dict containing the computed values of Precision, Recall and F1 metrics.
    """

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def compute_metrics(gt_actions: np.ndarray, pred_actions: np.ndarray):
    """Compute offline OpenVLA action prediction metrics.

    Expected action layout: [dx, dy, dz, drz, dry, drz, gripper]


    Args:
        gt_actions (np.ndarray): The ground-truth action the expert has taken.
        pred_actions (np.ndarray): The predicted actions from our model.
    """

    # Ensure the input format is correct.
    gt = np.asarray(gt_actions, dtype=np.float64)
    pred = np.asarray(pred_actions, dtype=np.float64)

    if gt.shape != pred.shape:
        raise ValueError(f"Action shape mismatch: GT={gt.shape}, Prediction={pred.shape}")

    if gt.ndim != 2 or gt.shape[1] != 7:
        raise ValueError(f"Expected actions with shape (N, 7), got {gt.shape}")

    if len(gt) == 0:
        raise ValueError("Cannot evaluate an empty set of actions.")

    if not np.isfinite(gt).all():
        raise ValueError("Ground truth contains NaN or Inf.")

    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain NaN or Inf.")

    # Grab the actions of the arm (we consider the gripper later...)
    gt_arm = gt[:, :6]
    pred_arm = pred[:, :6]

    # Compute the error between the action taken from the expert and our model.
    error = pred_arm - gt_arm
    abs_error = np.abs(error)

    axis_mae = {}
    for i, name in enumerate(ARM_AXIS_NAMES):
        axis_mae[name] = float(abs_error[:, i].mean())

    # Grab only the translation and rotation error separately
    translation_error = error[:, :3]
    rotation_error = error[:, 3:6]

    # Compute their MAE, RMSE, cosine similarity
    translation_mae = float(np.abs(translation_error).mean())
    translation_rmse = float(np.sqrt(np.mean(translation_error ** 2)))

    rotation_mae = float(np.abs(rotation_error).mean())
    rotation_rmse = float(np.sqrt(np.mean(rotation_error ** 2)))

    translation_vector_error = float(np.linalg.norm(translation_error, axis=1).mean())
    rotation_vector_error = float(np.linalg.norm(rotation_error, axis=1).mean())

    # Grab only the translation components
    gt_xyz = gt[:, :3]
    pred_xyz = pred[:, :3]

    gt_norm = np.linalg.norm(gt_xyz, axis=1)
    pred_norm = np.linalg.norm(pred_xyz, axis=1)

    # Check if the action has enough magnitude to not be considered a "zero-action"
    valid = (gt_norm > 1e-6) & (pred_norm > 1e-6)

    if np.any(valid):
        # If the norm on all axes are not considered zero-actions...
        dot_products = np.sum(gt_xyz[valid] * pred_xyz[valid], axis=1)
        cosine = dot_products / (gt_norm[valid] * pred_norm[valid])
        direction_cosine = float(np.clip(cosine, -1.0, 1.0).mean())
    else:
        direction_cosine = None

    # Zero-action metrics...
    # These indicate the average error we would make if we only predicted 0 every time. 
    zero_translation_mae = float(np.abs(gt[:, :3]).mean())
    zero_rotation_mae = float(np.abs(gt[:, 3:6]).mean())
    zero_arm_mae = float(np.abs(gt_arm).mean())

    arm_metrics = {
        "axis_mae": axis_mae,
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "translation_mae": translation_mae,
        "translation_rmse": translation_rmse,
        "translation_vector_error": translation_vector_error,
        "rotation_mae": rotation_mae,
        "rotation_rmse": rotation_rmse,
        "rotation_vector_error": rotation_vector_error,
        "direction_cosine": direction_cosine,
        "direction_cosine_valid_frames": int(valid.sum()),
        "zero_translation_mae": zero_translation_mae,
        "zero_rotation_mae": zero_rotation_mae,
        "zero_arm_mae": zero_arm_mae,
    }

    # Now onto the gripper metrics!
    # True = CLOSED; False = OPEN
    gt_closed = gt[:, 6] < 0.5
    pred_closed = pred[:, 6] < 0.5

    # Conventions: t = True, f = False, p = Positive, n = Negative
    # We make AND comparisons to check whether we predicted as the expert has done or not...
    # The ~ operator is just a NOT operator...
    tp = int(np.sum(gt_closed & pred_closed))
    tn = int(np.sum(~gt_closed & ~pred_closed))
    fp = int(np.sum(~gt_closed & pred_closed))
    fn = int(np.sum(gt_closed & ~pred_closed))

    # Call the Precision, Recall and F1 metrics computation...
    closed_metrics = classification_metrics(tp, fp, fn)

    # Switch positive class to OPEN.
    open_metrics = classification_metrics(tn, fn, fp)

    # We check the accuracy (number of time the prediction was the same as the expert action)
    accuracy = float(np.mean(gt_closed == pred_closed))

    # A more general F1 metric.
    macro_f1 = float((closed_metrics["f1"] + open_metrics["f1"]) / 2)

    gripper_metrics = {
        "accuracy": accuracy,
        "precision_closed": closed_metrics["precision"],
        "recall_closed": closed_metrics["recall"],
        "f1_closed": closed_metrics["f1"],
        "precision_open": open_metrics["precision"],
        "recall_open": open_metrics["recall"],
        "f1_open": open_metrics["f1"],
        "macro_f1": macro_f1,
        "num_closed_gt": int(gt_closed.sum()),
        "num_open_gt": int((~gt_closed).sum()),
        "confusion_matrix": [
            [tp, fn],
            [fp, tn],
        ],
    }

    return {
        "num_frames": len(gt),
        "arm": arm_metrics,
        "gripper": gripper_metrics,
    }


import json
from pathlib import Path


def save_metrics(model_name: str, dataset_dir: str, split_name: str, metrics: dict, results_dir: Path):

    results_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "dataset": dataset_dir,
        "split": split_name,
        "metrics": metrics
    }

    # Note: the extension is .jsonl, it is a variant of normal json, perfect for appending json objects independently!
    with open(results_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print(f"[SAVED] Results for {model_name}")

def generate_graphs():
    pass