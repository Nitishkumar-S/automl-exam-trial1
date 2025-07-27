import os
import json
from pathlib import Path
from training.trainer import train_model
from modeling.curve_model import extrapolate
from modeling.bo.bo_loop import suggest_next
from utils.config_sampling import sample_config
from utils.config_space import get_configspace
from tqdm import tqdm
import argparse

def run_bo_loop(n_init=5, n_iter=10, budget=10, out_dir="experiments/bo_runs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    config_list = []
    extrap_losses = []
    extrap_stds = []

    # Step 1: Bootstrap with random configs
    for i in tqdm(range(n_init), desc="Bootstrapping"):
        config = sample_config()
        result = train_model(config, budget, seed=i)
        val_losses = result["val_loss_curve"]
        observed_epochs = list(range(1, len(val_losses)+1))

        pred, std = extrapolate(config, val_losses, observed_epochs, target_epoch=50)

        config_list.append(config)
        extrap_losses.append(pred)
        extrap_stds.append(std)

        # Save run
        with open(Path(out_dir) / f"run_{i}.json", "w") as f:
            json.dump(result, f)

    # Step 2: BO iterations
    for i in tqdm(range(n_iter), desc="BO Loop"):
        next_config = suggest_next(config_list, extrap_losses, extrap_stds)
        result = train_model(next_config, budget, seed=1000+i)

        val_losses = result["val_loss_curve"]
        observed_epochs = list(range(1, len(val_losses)+1))
        pred, std = extrapolate(next_config, val_losses, observed_epochs, target_epoch=50)

        config_list.append(next_config)
        extrap_losses.append(pred)
        extrap_stds.append(std)

        # ✅ Add extrapolated prediction to result
        result["extrapolated_loss"] = pred
        result["extrapolated_std"] = std

        with open(Path(out_dir) / f"run_{n_init + i}.json", "w") as f:
            json.dump(result, f)

    # Select best config from surrogate estimates
    best_idx = np.argmin(extrap_losses)
    best_config = config_list[best_idx]

    print(f"\nBest predicted config: {best_config}")
    print(f"Training this config to full budget...")

    # Train it fully
    final_result = train_model(best_config, budget=50, seed=42)

    # Save final model info
    with open(Path(out_dir) / "final_best_config.json", "w") as f:
        json.dump({
            "config": best_config,
            "final_result": final_result
        }, f)

    print(f"Final validation accuracy: {final_result['val_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="experiments/bo_runs")
    args = parser.parse_args()

    run_bo_loop(n_init=args.n_init, n_iter=args.n_iter, budget=args.budget, out_dir=args.out_dir)

# Command to run the script:
# python pipeline/bo_loop.py \
#   --n_init 5 \
#   --n_iter 10 \
#   --budget 10 \
#   --out_dir experiments/bo_runs

python pipeline/bo_loop.py --n_init 5 --n_iter 10 --budget 10 --out_dir experiments/bo_runs