import optuna
import os
import argparse
import yaml

from utils import convert_optuna_params_to_config
from models.hyperparameter_tuning import objective
# Set a fixed seed for reproducibility
SEED = 42

def save_best_params(study, trial):
    """Callback function to save the best parameters after each trial."""
    current_best_params_path = os.path.join(".", "experiments", "current_trial_best.yaml")
    structured_current_best_params = convert_optuna_params_to_config(study.best_params)

    with open(current_best_params_path, "w") as f:
        yaml.dump(structured_current_best_params, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters using Optuna.")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of trials to run")
    parser.add_argument('--study_name', type=str, default="default", help="Optuna study name")
    parser.add_argument('--optuna_storage_path', type=str, default=os.path.join(".", "experiments", "optuna_studies", "study.db"), help="Storage path for persistent study database")
    parser.add_argument('--min_resource', type=int, default=10, help="Minimum resource for Hyperband Pruner")
    parser.add_argument('--max_resource', type=int, default=200, help="Maximum resource for Hyperband Pruner")
    parser.add_argument('--reduction_factor', type=int, default=4, help="Reduction factor for Hyperband Pruner")

    args = parser.parse_args()

    # Create storage path
    storage_path = f"sqlite:///{args.optuna_storage_path}"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(args.optuna_storage_path), exist_ok=True)

    # Uncomment to reset the database before each run
    # if os.path.exists(args.optuna_storage_path):
    #     os.remove(args.optuna_storage_path)
    #     print(f"Deleted existing study database at '{args.optuna_storage_path}'.")

    # Check write permissions
    if not os.access(os.path.dirname(args.optuna_storage_path), os.W_OK):
        raise PermissionError(f"Cannot write to '{args.optuna_storage_path}'. Check permissions.")

    # Define Optuna pruner
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource,
        max_resource=args.max_resource,
        reduction_factor=args.reduction_factor
    )

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=storage_path,
        load_if_exists=True  # Resume study if it exists
    )

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, timeout=2000, callbacks=[save_best_params])

    # Define the save path for the best study results
    best_params_path = os.path.join(".", "experiments", "trial_best.yaml")

    # Save best parameters to a YAML file
    structured_best_params = convert_optuna_params_to_config(study.best_params)

    with open(best_params_path, "w") as f:
        yaml.dump(structured_best_params, f, default_flow_style=False)

    print(f"\nBest hyperparameters saved to: {best_params_path}\n")

    # Print and log the best hyperparameters
    print("\nBest hyperparameters are:")
    print(study.best_params)



