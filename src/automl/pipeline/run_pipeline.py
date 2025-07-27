import json
from training.trainer import train_model
from utils.config_sampling import sample_config
from pathlib import Path
from tqdm import tqdm
import argparse
import uuid

def main(n_runs: int, budget: int, out_dir: str):
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(n_runs)):
        # Sample a random hyperparameter configuration
        config = sample_config()

        # Train the model using the sampled config and budget (epochs)
        result = train_model(config, budget=budget, seed=i)

        # Add a unique ID for tracking this run
        result["config_id"] = str(uuid.uuid4())

        # Save results (config + val loss curve + val acc curve) to JSON
        out_file = Path(out_dir) / f"result_{i}.json"
        with open(out_file, "w") as f:
            json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of configs to run")
    parser.add_argument("--budget", type=int, default=10, help="Training epochs per config")
    parser.add_argument("--out_dir", type=str, default="experiments/runs", help="Output folder for saving results")
    args = parser.parse_args()

    main(n_runs=args.n, budget=args.budget, out_dir=args.out_dir)
