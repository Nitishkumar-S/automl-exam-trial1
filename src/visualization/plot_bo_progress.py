import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_losses(path):
    files = sorted([f for f in os.listdir(path) if f.startswith("run_") and f.endswith(".json")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))

    pred_losses = []
    for file in files:
        with open(os.path.join(path, file)) as f:
            data = json.load(f)
            pred = data.get("extrapolated_loss", None)  # optional
            if pred is None:
                # fallback: use final val loss
                pred = data["val_loss_curve"][-1]
            pred_losses.append(pred)
    return pred_losses

def plot_curve(losses, out_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', label="Predicted loss")
    plt.xlabel("BO Iteration")
    plt.ylabel("Validation Loss (extrapolated or final)")
    plt.title("BO Progress Over Iterations")
    plt.grid(True)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="experiments/bo_runs")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    losses = load_losses(args.exp_dir)
    plot_curve(losses, args.save_path)


# Command to run the script:
# python src/visualization/plot_bo_progress.py --exp_dir experiments/bo_runs --save_path plots/bo_progress.png
# This will plot the BO progress and save it to the specified path or display it if no save path is provided.