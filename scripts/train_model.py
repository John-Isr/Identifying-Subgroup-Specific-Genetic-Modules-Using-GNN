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
                        help="Path to the hyper parameters YAML config file.")
    parser.add_argument("--data_dir",
                        type=str,
                        default=os.path.join(os.getcwd(),"data", "modified_graphs"),
                        help="Directory containing processed graph data.")
    parser.add_argument("--epochs",
                        type=int,
                        default=250,
                        help="Number of training epochs (not in the config).")

    return parser.parse_args()

def main():
    args = parse_arguments()

    ## Load & Flatten Config
    config = load_config(args.config)  # Nested YAML
    hparams = compile_hyperparams_from_config(config)  # Flatten

    ## Get params from CLI
    data_dir = args.data_dir
    num_epochs = args.epochs

    batch_size = hparams["batch_size"]
    node_embedding_type = hparams["node_embedding_type"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Training for {num_epochs} epochs")
    print(f"Hparams: {hparams}")
    print(f"Data directory: {data_dir}")

    ## Initialize W&B
    init_wandb(hparams)

    ## Load Data

    train_loader, val_loader,test_loader = load_data_splits(batch_size,node_embedding_type=node_embedding_type,graph_dir=data_dir)

    ## Build the Model
    # Get a sample batch to determine input_dim & edge_dim
    batch = next(iter(val_loader))
    graphs = batch.to_data_list()  # Use PyG's built-in function
    # Access a specific graph from the batch
    graph = graphs[0]
    input_dim = graph.x.size(1)  # Number of node feature dimensions
    edge_dim = graph.edge_attr.size(1)  # Number of edge feature dimensions

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

        scheduler_args = {}
        for key in ("T_max", "eta_min", "factor", "patience", "step_size",
                    "gamma", "base_lr", "max_lr", "step_size_up", "mode"):
            if key in hparams:
                scheduler_args[key] = hparams[key]

        scheduler = scheduler_class(optimizer, **scheduler_args)

    ## Train the Model
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

    ## Evaluate on Test Set
    test_metrics = evaluate_model(trained_model, test_loader)
    f1_score = test_metrics["f1_score"]
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]

    print(f"\nTest Metrics:\n"
          f"  F1 Score:   {f1_score:.4f}\n"
          f"  Precision:  {precision:.4f}\n"
          f"  Recall:     {recall:.4f}")

    # Log results
    log_test_metrics(f1_score, precision, recall)
    finish_logging()

    ## Save the Trained Model
    model_save_path = os.path.join(".", "experiments", "trained_model.pt")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
