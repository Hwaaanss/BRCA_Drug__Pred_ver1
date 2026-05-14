"""Small plotting helpers for training scripts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_curves(loss_histories, plot_dir, plot_filename="loss_curves.png", title="Training Loss"):
    """Save per-fold train/validation loss curves to plot_dir/plot_filename."""
    if not loss_histories:
        return None

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for history in loss_histories:
        epochs = history.get("epoch", [])
        label_suffix = f"fold {history.get('fold', '?')}"
        if history.get("train_loss"):
            ax.plot(epochs, history["train_loss"], alpha=0.8, label=f"train {label_suffix}")
        if history.get("val_loss"):
            ax.plot(epochs, history["val_loss"], alpha=0.8, linestyle="--", label=f"val {label_suffix}")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = plot_dir / plot_filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
