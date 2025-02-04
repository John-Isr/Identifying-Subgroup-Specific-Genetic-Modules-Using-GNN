# todo: implement the main training logic here
#!/usr/bin/env python3
from utils import load_data_splits, compile_hyperparams_from_config
from utils.logging import log_test_metrics, finish_logging, init_wandb
from models.architecture import GNNClassifier
from models.training import train_model, evaluate_model

import os
import gc
import yaml
import argparse
import torch
import torch.nn as nn


def load_config(config_path: str) -> dict:
    """Loads a nested YAML config from file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GNN model with a given config.")
    parser.add_argument("--config",
                        type=str,
                        default=os.path.join("experiments", "default_config.yaml"),
                        help="Path to the YAML config file (structured).")
    parser.add_argument("--data_dir",
                        type=str,
                        default=os.path.join("data", "modified_graphs"),
                        help="Directory containing processed graph data.")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs (not in the config).")
    parser.add_argument("--batch_size",
                        type=int,
                        default=None,
                        help="Batch size (override flattened config if specified).")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cpu or cuda).")

    return parser.parse_args()

def main():
    args = parse_arguments()

    ## Load & Flatten Config
    config = load_config(args.config)  # Nested YAML
    hparams = compile_hyperparams_from_config(config)  # Flatten

    ## Override from CLI
    if args.batch_size is not None:
        hparams["batch_size"] = args.batch_size
    data_dir = args.data_dir
    device = args.device
    num_epochs = args.epochs

    print(f"Using device: {device}")
    print(f"Training for {num_epochs} epochs")
    print(f"Hparams: {hparams}")

    ## Initialize W&B
    init_wandb(hparams)

    ## Load Data

    train_loader, val_loader, test_loader = load_data_splits(
        batch_size=hparams["batch_size"],device=device,
        node_embedding_type=hparams["node_embedding_type"],
        graph_dir=data_dir
    )

    ## Build the Model
    # Get a sample batch to determine input_dim & edge_dim
    first_batch = next(iter(train_loader))
    graphs = first_batch.to_data_list()
    sample_graph = graphs[0]
    input_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)

    model = GNNClassifier(
        input_dim=input_dim,
        edge_dim=edge_dim,
        num_layers=hparams["num_layers"],
        hidden_dim=hparams["hidden_dim"],
        heads=hparams["heads"],
        dropout=hparams["dropout"]
    ).to(device)

    ## Define Loss, Optimizer, Scheduler
    pos_weight = torch.tensor([hparams["pos_weight_ratio"]], dtype=torch.float32).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer_name = hparams["optimizer"]
    lr = hparams["lr"]
    weight_decay = hparams["weight_decay"]

    # Build optimizer
    optimizer_params = {}
    if optimizer_name == "RMSprop":
        # Fill RMSProp-specific params if present
        if "rmsprop_momentum" in hparams:
            optimizer_params["momentum"] = hparams["rmsprop_momentum"]
        if "rmsprop_alpha" in hparams:
            optimizer_params["alpha"] = hparams["rmsprop_alpha"]

    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_params
    )

    # Build scheduler if not 'none'
    scheduler = None
    scheduler_name = hparams["scheduler"]
    if scheduler_name != "none":
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)

        # We pass the relevant hparams. For example:
        #   if scheduler_name == "CosineAnnealingLR", we expect "T_max" and "eta_min"
        #   if "CyclicLR", we expect base_lr, max_lr, step_size_up, mode, etc.
        scheduler_args = {}
        # We'll just copy from hparams if it exists
        # e.g. "T_max" in hparams means scheduler_args["T_max"] = hparams["T_max"]
        # but let's do it conditionally:
        for key in ("T_max", "eta_min", "factor", "patience", "step_size",
                    "gamma", "base_lr", "max_lr", "step_size_up", "mode"):
            if key in hparams:
                scheduler_args[key] = hparams[key]

        # For ReduceLROnPlateau specifically, we included "mode": "max" in hparams
        scheduler = scheduler_class(optimizer, **scheduler_args)

    ## Train the Model

    try:
        trained_model = train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            trial=None  # Not using Optuna trial here
        )
    except RuntimeError as e:
        # Graceful handling of OOM or other issues
        if "out of memory" in str(e).lower():
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print("Encountered CUDA OOM. Training aborted.")
            return
        else:
            raise e

    ## Evaluate on Test Set
    test_metrics = evaluate_model(trained_model, test_loader)
    f1_score = test_metrics["f1_score"]
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]

    print(f"\nTest Metrics:\n"
          f"  F1 Score:   {f1_score:.4f}\n"
          f"  Precision:  {precision:.4f}\n"
          f"  Recall:     {recall:.4f}")

    ## Optional W&B logging:
    log_test_metrics(f1_score, precision, recall)
    finish_logging()

    ## Save the Trained Model (optional)
    model_save_path = os.path.join(".", "experiments", "trained_model.pt")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
