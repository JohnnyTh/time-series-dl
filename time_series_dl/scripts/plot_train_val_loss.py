import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss_history(
    json_path: str | Path,
    save_path: str | Path = None,
    title: str = None,
):
    """
    Plot training and validation loss curves from JSON history.

    Parameters
    ----------
    json_path : str or Path
        Path to JSON file with keys: "train_loss", "val_loss"

    save_path : str or Path, optional
        Output path (e.g. .png or .pdf). If None, uses json_path stem.

    title : str, optional
        Custom plot title
    """

    json_path = Path(json_path)

    with open(json_path, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if len(train_loss) == 0:
        raise ValueError("train_loss is empty")

    epochs = list(range(1, len(train_loss) + 1))

    # --- figure setup ---
    plt.figure(figsize=(8, 4.5))  # good for thesis (fits page width)

    plt.plot(epochs, train_loss, label="Training loss", linewidth=2)

    if len(val_loss) > 0:
        plt.plot(epochs[: len(val_loss)], val_loss, label="Validation loss", linewidth=2)

    # --- formatting ---
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")

    if title is None:
        title = f"Training and Validation Loss ({json_path.stem})"

    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # --- save ---
    if save_path is None:
        save_path = json_path.with_suffix(".png")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_loss_history(
        json_path="results/dl_models/NBEATS_loss_history.json",
        save_path="figures/nbeats_loss.png",
        title="N-BEATS Training and Validation Loss",
    )
