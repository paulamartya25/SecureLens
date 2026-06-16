"""
utils/confusion_matrix.py
SecureLens — Confusion Matrix, ROC Curve, Classification Report

Run:
    python utils/confusion_matrix.py

Outputs saved to:
    docs/confusion_matrix.png
    docs/roc_curve.png
    docs/training_history.png
    docs/classification_report.txt
    docs/model_metrics.json
"""

import os
import sys
import json

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # Saves images without opening GUI window

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "cloud_server", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

os.makedirs(DOCS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_percentage(numerator, denominator):
    """
    Avoid division-by-zero error.
    """
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------

def load_model():
    """
    Loads the trained SecureLens model and preprocessing transform.
    """

    from cloud_server.train_model import SecureLensNet
    from torchvision import transforms

    model_path = os.path.join(MODELS_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected file: cloud_server/models/best_model.pth"
        )

    print(f"[Model] Loading model from: {model_path}")

    model = SecureLensNet(num_classes=2)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return model, tf


# ---------------------------------------------------------------------
# Get predictions on test dataset
# ---------------------------------------------------------------------

def get_predictions(model, tf):
    """
    Runs inference on the test dataset and returns:
        labels, predictions, probabilities
    """

    from torch.utils.data import DataLoader
    from cloud_server.train_model import ChestXRayDataset

    test_dir = os.path.join(DATA_DIR, "test")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Test dataset folder not found: {test_dir}\n"
            f"Expected structure: data/chest_xray/test/NORMAL and PNEUMONIA"
        )

    print("[CM] Loading test dataset...")

    test_ds = ChestXRayDataset(DATA_DIR, "test", tf)

    loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    all_preds = []
    all_labels = []
    all_probs = []

    print("[CM] Running inference on test set...")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ---------------------------------------------------------------------
# Plot and save confusion matrix
# ---------------------------------------------------------------------

def plot_confusion_matrix(labels, preds, save_path):
    """
    Creates and saves:
        Confusion Matrix + Classification Metrics chart
    """

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    sensitivity = safe_percentage(tp, tp + fn)
    specificity = safe_percentage(tn, tn + fp)
    ppv = safe_percentage(tp, tp + fp)
    npv = safe_percentage(tn, tn + fn)
    accuracy = safe_percentage(tp + tn, tp + tn + fp + fn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # --------------------------------------------------------------
    # Confusion matrix heatmap
    # --------------------------------------------------------------

    ax = axes[0]
    ax.set_facecolor("#1a1f2e")

    im = ax.imshow(cm, cmap="Greys_r", aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Count", color="white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Pneumonia"],
        color="white",
        fontsize=12
    )

    ax.set_yticklabels(
        ["Normal", "Pneumonia"],
        color="white",
        fontsize=12
    )

    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_ylabel("True Label", color="white", fontsize=12)

    ax.set_title(
        "Confusion Matrix",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    ax.tick_params(colors="white")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="#e2e8f0",
                fontsize=18,
                fontweight="bold"
            )

    # --------------------------------------------------------------
    # Metrics bar chart
    # --------------------------------------------------------------

    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")

    metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
    }

    colors = [
        "#63b3ed",
        "#68d391",
        "#f6e05e",
        "#fc8181",
        "#b794f4",
    ]

    bars = ax2.barh(
        list(metrics.keys()),
        list(metrics.values()),
        color=colors,
        height=0.5
    )

    ax2.set_xlim(0, 115)

    ax2.set_xlabel("Percentage (%)", color="white", fontsize=12)

    ax2.set_title(
        "Classification Metrics",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    ax2.tick_params(colors="white")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    for bar, val in zip(bars, metrics.values()):
        ax2.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout(pad=2)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="#0f1117"
    )

    plt.close()

    print(f"[CM] Confusion matrix saved → {save_path}")

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "sensitivity": round(sensitivity, 2),
        "specificity": round(specificity, 2),
        "ppv": round(ppv, 2),
        "npv": round(npv, 2),
        "accuracy": round(accuracy, 2),
    }


# ---------------------------------------------------------------------
# Plot and save ROC + Precision-Recall curves
# ---------------------------------------------------------------------

def plot_roc_curve(labels, probs, save_path):
    """
    Creates and saves:
        ROC curve + Precision-Recall curve
    """

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # --------------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------------

    ax = axes[0]
    ax.set_facecolor("#1a1f2e")

    ax.plot(
        fpr,
        tpr,
        color="#63b3ed",
        lw=2,
        label=f"ROC AUC = {roc_auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="#4a5568",
        lw=1,
        linestyle="--",
        label="Random"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")

    ax.set_title(
        "ROC Curve — Pneumonia Detection",
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax.tick_params(colors="white")

    ax.legend(
        loc="lower right",
        facecolor="#2d3748",
        labelcolor="white"
    )

    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    # --------------------------------------------------------------
    # Precision-Recall curve
    # --------------------------------------------------------------

    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")

    ax2.plot(
        recall,
        precision,
        color="#68d391",
        lw=2,
        label=f"Average Precision = {pr_auc:.4f}"
    )

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])

    ax2.set_xlabel("Recall", color="white")
    ax2.set_ylabel("Precision", color="white")

    ax2.set_title(
        "Precision-Recall Curve",
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax2.tick_params(colors="white")

    ax2.legend(
        loc="lower left",
        facecolor="#2d3748",
        labelcolor="white"
    )

    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    plt.tight_layout(pad=2)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="#0f1117"
    )

    plt.close()

    print(f"[ROC] ROC + PR curves saved → {save_path}")

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


# ---------------------------------------------------------------------
# Plot and save training history
# ---------------------------------------------------------------------

def plot_training_history(save_path):
    """
    Creates and saves:
        Training accuracy/loss curves
    """

    hist_path = os.path.join(MODELS_DIR, "training_history.json")

    if not os.path.exists(hist_path):
        print("[History] No training history found — skipping.")
        return

    with open(hist_path, "r") as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_acc"]) + 1))

    train_acc = [x * 100 for x in h["train_acc"]]
    val_acc = [x * 100 for x in h["val_acc"]]

    train_loss = h["train_loss"]
    val_loss = h["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # --------------------------------------------------------------
    # Accuracy curve
    # --------------------------------------------------------------

    ax = axes[0]
    ax.set_facecolor("#1a1f2e")

    ax.plot(
        epochs,
        train_acc,
        color="#63b3ed",
        lw=2,
        label="Train",
        marker="o",
        markersize=3
    )

    ax.plot(
        epochs,
        val_acc,
        color="#68d391",
        lw=2,
        label="Validation",
        marker="s",
        markersize=3,
        linestyle="--"
    )

    ax.axhline(
        y=89.42,
        color="#fc8181",
        lw=1,
        linestyle=":",
        label="Test 89.42%"
    )

    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Accuracy %", color="white")

    ax.set_title(
        "Training vs Validation Accuracy",
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax.tick_params(colors="white")

    ax.legend(
        facecolor="#2d3748",
        labelcolor="white"
    )

    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    # --------------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------------

    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")

    ax2.plot(
        epochs,
        train_loss,
        color="#63b3ed",
        lw=2,
        label="Train",
        marker="o",
        markersize=3
    )

    ax2.plot(
        epochs,
        val_loss,
        color="#68d391",
        lw=2,
        label="Validation",
        marker="s",
        markersize=3,
        linestyle="--"
    )

    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Loss", color="white")

    ax2.set_title(
        "Training vs Validation Loss",
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax2.tick_params(colors="white")

    ax2.legend(
        facecolor="#2d3748",
        labelcolor="white"
    )

    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    plt.tight_layout(pad=2)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="#0f1117"
    )

    plt.close()

    print(f"[History] Training curves saved → {save_path}")


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  SecureLens — Model Evaluation")
    print("=" * 55)

    model, tf = load_model()

    labels, preds, probs = get_predictions(model, tf)

    print("\n[Report] Classification Report:")

    report = classification_report(
        labels,
        preds,
        target_names=["Normal", "Pneumonia"],
        zero_division=0
    )

    print(report)

    report_path = os.path.join(DOCS_DIR, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SecureLens — Classification Report\n")
        f.write("=" * 40 + "\n")
        f.write(report)

    print(f"[Report] Saved → {report_path}")

    cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(DOCS_DIR, "roc_curve.png")
    hist_path = os.path.join(DOCS_DIR, "training_history.png")

    cm_results = plot_confusion_matrix(labels, preds, cm_path)

    roc_results = plot_roc_curve(labels, probs, roc_path)

    plot_training_history(hist_path)

    all_metrics = {
        **cm_results,
        **roc_results,
        "test_accuracy": cm_results["accuracy"],
    }

    metrics_path = os.path.join(DOCS_DIR, "model_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[Metrics] Saved → {metrics_path}")

    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)

    print(f"  Test Accuracy   : {cm_results['accuracy']}%")
    print(f"  Sensitivity     : {cm_results['sensitivity']}%")
    print(f"  Specificity     : {cm_results['specificity']}%")
    print(f"  ROC AUC         : {roc_results['roc_auc']}")
    print(f"  PR AUC          : {roc_results['pr_auc']}")

    print("\n  Saved output files:")
    print(f"  - {cm_path}")
    print(f"  - {roc_path}")
    print(f"  - {hist_path}")
    print(f"  - {report_path}")
    print(f"  - {metrics_path}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()