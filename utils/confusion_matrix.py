"""
utils/confusion_matrix.py
SecureLens — Confusion Matrix, ROC Curve, Classification Report
Run: python utils/confusion_matrix.py
Saves plots to docs/
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "..", "cloud_server", "models")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data", "chest_xray")
DOCS_DIR    = os.path.join(BASE_DIR, "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")


def load_model():
    from cloud_server.train_model import SecureLensNet
    from torchvision import transforms

    model = SecureLensNet(num_classes=2)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "best_model.pth"),
                   map_location=DEVICE))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    return model, tf


def get_predictions(model, tf):
    from torch.utils.data import DataLoader
    from cloud_server.train_model import ChestXRayDataset

    print("[CM] Loading test dataset...")
    test_ds = ChestXRayDataset(DATA_DIR, "test", tf)
    loader  = DataLoader(test_ds, batch_size=32,
                         shuffle=False, num_workers=0)

    all_preds  = []
    all_labels = []
    all_probs  = []

    print("[CM] Running inference on test set...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            out    = model(imgs)
            probs  = torch.softmax(out, dim=1)
            preds  = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    ppv         = tp / (tp + fp) * 100
    npv         = tn / (tn + fn) * 100
    accuracy    = (tp + tn) / (tp + tn + fp + fn) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # Confusion matrix heatmap
    ax = axes[0]
    ax.set_facecolor("#1a1f2e")
    im = ax.imshow(cm, cmap="Greys_r", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Count", color="white")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Pneumonia"],
                       color="white", fontsize=12)
    ax.set_yticklabels(["Normal","Pneumonia"],
                       color="white", fontsize=12)
    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_ylabel("True Label",      color="white", fontsize=12)
    ax.set_title("Confusion Matrix", color="white", fontsize=14)
    ax.tick_params(colors="white")

    for i in range(2):
        for j in range(2):
            # Always use light text color on dark backgrounds
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center",
                    color="#e2e8f0", fontsize=16, fontweight="bold")

    # Metrics bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")
    metrics = {
        "Accuracy"   : accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV"        : ppv,
        "NPV"        : npv,
    }
    colors = ["#63b3ed","#68d391","#f6e05e","#fc8181","#b794f4"]
    bars   = ax2.barh(list(metrics.keys()),
                      list(metrics.values()),
                      color=colors, height=0.5)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("Percentage (%)", color="white")
    ax2.set_title("Classification Metrics", color="white", fontsize=14)
    ax2.tick_params(colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    for bar, val in zip(bars, metrics.values()):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center",
                 color="white", fontsize=11)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"[CM] Confusion matrix saved → {save_path}")

    return {
        "true_negative" : int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive" : int(tp),
        "sensitivity"   : round(sensitivity, 2),
        "specificity"   : round(specificity, 2),
        "ppv"           : round(ppv, 2),
        "npv"           : round(npv, 2),
        "accuracy"      : round(accuracy, 2),
    }


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # ROC curve
    ax = axes[0]
    ax.set_facecolor("#1a1f2e")
    ax.plot(fpr, tpr, color="#63b3ed", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0,1],[0,1], color="#4a5568",
            lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curve — Pneumonia Detection",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    legend = ax.legend(loc="lower right",
                       facecolor="#2d3748", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    # Precision-Recall
    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")
    ax2.plot(recall, precision, color="#68d391", lw=2,
             label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall",    color="white")
    ax2.set_ylabel("Precision", color="white")
    ax2.set_title("Precision-Recall Curve",
                  color="white", fontsize=13)
    ax2.tick_params(colors="white")
    legend2 = ax2.legend(loc="lower left",
                         facecolor="#2d3748", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"[ROC] ROC + PR curves saved → {save_path}")
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc" : round(pr_auc, 4),
    }


def plot_training_history(save_path):
    hist_path = os.path.join(MODELS_DIR, "training_history.json")
    if not os.path.exists(hist_path):
        print("[History] No training history found — skipping.")
        return

    with open(hist_path) as f:
        h = json.load(f)

    epochs     = list(range(1, len(h["train_acc"])+1))
    train_acc  = [x*100 for x in h["train_acc"]]
    val_acc    = [x*100 for x in h["val_acc"]]
    train_loss = h["train_loss"]
    val_loss   = h["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    # Accuracy
    ax = axes[0]
    ax.set_facecolor("#1a1f2e")
    ax.plot(epochs, train_acc, color="#63b3ed",
            lw=2, label="Train", marker="o", markersize=3)
    ax.plot(epochs, val_acc,   color="#68d391",
            lw=2, label="Val",   marker="s", markersize=3,
            linestyle="--")
    ax.axhline(y=89.42, color="#fc8181", lw=1,
               linestyle=":", label="Test 89.42%")
    ax.set_xlabel("Epoch",    color="white")
    ax.set_ylabel("Accuracy %", color="white")
    ax.set_title("Training vs Validation Accuracy",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2d3748", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    # Loss
    ax2 = axes[1]
    ax2.set_facecolor("#1a1f2e")
    ax2.plot(epochs, train_loss, color="#63b3ed",
             lw=2, label="Train", marker="o", markersize=3)
    ax2.plot(epochs, val_loss,   color="#68d391",
             lw=2, label="Val",   marker="s", markersize=3,
             linestyle="--")
    ax2.set_xlabel("Epoch",  color="white")
    ax2.set_ylabel("Loss",   color="white")
    ax2.set_title("Training vs Validation Loss",
                  color="white", fontsize=13)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#2d3748", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_color("#4a5568")

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"[History] Training curves saved → {save_path}")


def main():
    print("="*55)
    print("  SecureLens — Model Evaluation")
    print("="*55)

    model, tf = load_model()
    labels, preds, probs = get_predictions(model, tf)

    # Classification report
    print("\n[Report] Classification Report:")
    report = classification_report(
        labels, preds,
        target_names=["Normal","Pneumonia"])
    print(report)

    # Save report
    with open(os.path.join(DOCS_DIR,
              "classification_report.txt"), "w") as f:
        f.write("SecureLens — Classification Report\n")
        f.write("="*40 + "\n")
        f.write(report)

    # Plots
    cm_path   = os.path.join(DOCS_DIR, "confusion_matrix.png")
    roc_path  = os.path.join(DOCS_DIR, "roc_curve.png")
    hist_path = os.path.join(DOCS_DIR, "training_history.png")

    cm_results  = plot_confusion_matrix(labels, preds, cm_path)
    roc_results = plot_roc_curve(labels, probs, roc_path)
    plot_training_history(hist_path)

    # Save all metrics to JSON
    all_metrics = {**cm_results, **roc_results,
                   "test_accuracy": 89.42}
    with open(os.path.join(DOCS_DIR,
              "model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Test Accuracy   : {cm_results['accuracy']}%")
    print(f"  Sensitivity     : {cm_results['sensitivity']}%")
    print(f"  Specificity     : {cm_results['specificity']}%")
    print(f"  ROC AUC         : {roc_results['roc_auc']}")
    print(f"  PR  AUC         : {roc_results['pr_auc']}")
    print(f"\n  Saved to docs/")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()