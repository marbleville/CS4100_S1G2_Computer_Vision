"""
Evaluate trained GestureCNN on the test set.

Loads the best checkpoint, runs inference on held-out test data,
and produces accuracy, per-class precision/recall/F1, and a
confusion matrix saved to artifacts/eval/.

Usage:
    python3 classifier/evaluate.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from classifier.config import (
    HAND_CROP_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    STATIC_GESTURE_CLASSES,
)
from classifier.models.cnn import GestureCNN
from classifier.data.splits import load_splits, get_split_paths
from classifier.train import GestureTrainDataset

# Configuration                                                       
CHECKPOINT_PATH = "artifacts/models/cnn_best.pt"
MANIFEST_PATH = "data/manifest_split.csv"
OUTPUT_DIR = "artifacts/eval"
BATCH_SIZE = 64
NUM_WORKERS = 0

# Evaluation                                                           
def evaluate() -> dict:
    """
    Run evaluation on the test set using the best saved checkpoint.

    Returns:
        dict containing accuracy, per-class precision/recall/F1,
        and the confusion matrix as a numpy array.
    """
    print("=" * 60)
    print("GestureCNN Evaluation")
    print("=" * 60)

    # Load test split
    print("\nLoading test split...")
    splits = load_splits(MANIFEST_PATH)
    test_samples = get_split_paths(splits, "test")
    print(f"  Test samples: {len(test_samples)}")

    test_dataset = GestureTrainDataset(test_samples, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = GestureCNN(num_classes=len(STATIC_GESTURE_CLASSES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Checkpoint loaded from {CHECKPOINT_PATH}")

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    num_classes = len(STATIC_GESTURE_CLASSES)
    accuracy = (all_preds == all_labels).mean()

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    # Per-class precision, recall, F1
    per_class = {}
    for i, cls in enumerate(STATIC_GESTURE_CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(confusion[i, :].sum()),
        }

    # Macro averages
    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"{'=' * 60}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 56)
    for cls, metrics in per_class.items():
        print(
            f"{cls:<14} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>10}"
        )
    print("-" * 56)
    print(
        f"{'macro avg':<14} "
        f"{macro_precision:>10.4f} "
        f"{macro_recall:>10.4f} "
        f"{macro_f1:>10.4f} "
        f"{len(test_samples):>10}"
    )

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>10}" + "".join(f"{cls:>10}" for cls in STATIC_GESTURE_CLASSES)
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(STATIC_GESTURE_CLASSES):
        row = f"{cls:>10}" + "".join(f"{confusion[i][j]:>10}" for j in range(num_classes))
        print(row)

    # Save confusion matrix plot
    _save_confusion_matrix_plot(confusion, OUTPUT_DIR)

    # Save metrics to JSON
    import json
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    with open(output_path / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/eval_results.json")

    return results


def _save_confusion_matrix_plot(
    confusion: np.ndarray,
    output_dir: str,
) -> None:
    """
    Save a confusion matrix as a PNG using matplotlib.
    Falls back gracefully if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        fig.colorbar(im)

        ax.set_xticks(range(len(STATIC_GESTURE_CLASSES)))
        ax.set_yticks(range(len(STATIC_GESTURE_CLASSES)))
        ax.set_xticklabels(STATIC_GESTURE_CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(STATIC_GESTURE_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — Test Set")

        # Annotate cells with counts
        thresh = confusion.max() / 2
        for i in range(len(STATIC_GESTURE_CLASSES)):
            for j in range(len(STATIC_GESTURE_CLASSES)):
                ax.text(
                    j, i, str(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black",
                )

        fig.tight_layout()
        output_path = Path(output_dir) / "cnn_confusion_matrix.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {output_path}")

    except ImportError:
        print("matplotlib not installed — skipping confusion matrix plot.")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    evaluate()