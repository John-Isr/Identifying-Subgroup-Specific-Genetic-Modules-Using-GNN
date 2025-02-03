import wandb
import torch
from functools import wraps
import os

def ensure_wandb_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wandb.run:
            wandb.init(project="GNN-Modules")
        return func(*args, **kwargs)
    return wrapper

def finish_logging():
    """
    Finish logging the current run.
    """
    wandb.finish()

def init_wandb(config):
    """
    Initialize Weights and Biases with the given configuration.
    :param config: a dictionary containing the configuration for the run.
    """
    os.environ["WANDB_DISABLE_SYSTEM"] = "true"
    wandb.init(project="GNN-Modules", config=config,settings=wandb.Settings(x_disable_stats=True, console="off", disable_git=True))


def convert_optuna_params_to_config(best_params: dict) -> dict:
    """
    Convert the flat Optuna best_params dictionary into a structured config
    matching the parameter space used during the trials.

    This makes it easier to load these hyperparameters directly in your
    training scripts.
    """

    # --- Extract top-level (training) parameters ---
    batch_size = best_params["batch_size"]
    pos_weight_ratio = best_params["pos_weight_ratio"]

    # --- Model Architecture ---
    num_layers = best_params["num_layers"]
    hidden_dim = best_params["hidden_dim"]
    heads = best_params["heads"]
    dropout = best_params["dropout"]
    node_embedding = best_params["node_embedding"]  # "weighted", "spectral_positional_encoding", or "unweighted"

    # --- Optimizer Configuration ---
    optimizer_name = best_params["optimizer"]
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]

    # RMSProp-specific parameters (only present if optimizer == "RMSprop")
    optimizer_params = {}
    if optimizer_name == "RMSprop":
        optimizer_params["momentum"] = best_params["rmsprop_momentum"]
        optimizer_params["alpha"] = best_params["rmsprop_alpha"]

    # --- Scheduler Configuration ---
    scheduler_name = best_params["scheduler"]
    scheduler_params = {}

    # Only populate scheduler_params if scheduler is not 'none'
    if scheduler_name != "none":
        if scheduler_name == "CosineAnnealingLR":
            scheduler_params["T_max"] = best_params["T_max"]
            scheduler_params["eta_min"] = best_params["eta_min"]

        elif scheduler_name == "ReduceLROnPlateau":
            scheduler_params["factor"] = best_params["reduce_factor"]
            scheduler_params["patience"] = best_params["reduce_patience"]
            scheduler_params["mode"] = "max"  # Because you track F1

        elif scheduler_name == "StepLR":
            scheduler_params["step_size"] = best_params["step_size"]
            scheduler_params["gamma"] = best_params["step_gamma"]

        elif scheduler_name == "ExponentialLR":
            scheduler_params["gamma"] = best_params["exp_gamma"]

        elif scheduler_name == "CyclicLR":
            scheduler_params["base_lr"] = lr   # Use the chosen lr as the base
            scheduler_params["max_lr"] = best_params["cyclic_max_lr"]
            scheduler_params["step_size_up"] = best_params["cyclic_step_size_up"]
            scheduler_params["mode"] = best_params["cyclic_mode"]

    # --- Build a nested config dictionary ---
    structured_config = {
        "training": {
            "batch_size": batch_size,
            "pos_weight_ratio": pos_weight_ratio,
            "node_embedding": node_embedding,  # matches the param name 'node_embedding'

            "model_params": {
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "heads": heads,
                "dropout": dropout
            },

            # Optimizer Info
            "optimizer": optimizer_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "optimizer_params": optimizer_params,  # empty if not RMSprop

            # Scheduler Info
            "scheduler": scheduler_name,
            "scheduler_params": scheduler_params  # empty if 'none'
        }
    }

    return structured_config


def compile_hyperparams_from_config(config: dict) -> dict:
    """
    Convert a nested YAML config dictionary into a flat dict of hyperparameters,
    mirroring the structure returned by compile_hyperparams_from_trial.

    Assumes the config has a layout similar to:

    {
      "training": {
        "batch_size": 32,
        "pos_weight_ratio": 15,
        "node_embedding": "weighted",  # or "spectral_positional_encoding", "unweighted"
        "model_params": {
          "num_layers": 2,
          "hidden_dim": 128,
          "heads": 4,
          "dropout": 0.2
        },
        "optimizer": "RMSprop",  # or "Adam", "AdamW", etc.
        "optimizer_params": {
          "momentum": 0.9,
          "alpha": 0.95
        },
        "lr": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR",  # or "StepLR", "ReduceLROnPlateau", etc.
        "scheduler_params": {
          "T_max": 100,
          "eta_min": 1e-6
        }
      }
    }
    """

    training_cfg = config.get("training", {})

    # Pull out top-level fields
    batch_size = training_cfg.get("batch_size", 32)
    pos_weight_ratio = training_cfg.get("pos_weight_ratio", 1)
    node_embedding_type = training_cfg.get("node_embedding", "weighted")

    # Model params
    model_params = training_cfg.get("model_params", {})
    num_layers = model_params.get("num_layers", 2)
    hidden_dim = model_params.get("hidden_dim", 128)
    heads = model_params.get("heads", 4)
    dropout = model_params.get("dropout", 0.2)

    # Optimizer and scheduler
    optimizer = training_cfg.get("optimizer", "Adam")
    optimizer_params = training_cfg.get("optimizer_params", {})
    lr = training_cfg.get("lr", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-4)

    scheduler = training_cfg.get("scheduler", "none")
    scheduler_params = training_cfg.get("scheduler_params", {})

    hyperparams = {
        "batch_size": batch_size,
        "pos_weight_ratio": pos_weight_ratio,
        "node_embedding_type": node_embedding_type,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "heads": heads,
        "dropout": dropout,
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
    }

    # If RMSProp is used, add momentum/alpha if they exist
    if optimizer == "RMSprop":
        rms_momentum = optimizer_params.get("momentum")
        rms_alpha = optimizer_params.get("alpha")
        if rms_momentum is not None:
            hyperparams["rmsprop_momentum"] = rms_momentum
        if rms_alpha is not None:
            hyperparams["rmsprop_alpha"] = rms_alpha

    # If scheduler != "none", add its specific parameters
    if scheduler != "none":
        # CosineAnnealingLR
        if scheduler == "CosineAnnealingLR":
            hyperparams["T_max"] = scheduler_params.get("T_max")
            hyperparams["eta_min"] = scheduler_params.get("eta_min")

        elif scheduler == "ReduceLROnPlateau":
            hyperparams["factor"] = scheduler_params.get("factor")
            hyperparams["patience"] = scheduler_params.get("patience")
            # Hard-coded "mode" because F1 is max
            hyperparams["mode"] = "max"

        elif scheduler == "StepLR":
            hyperparams["step_size"] = scheduler_params.get("step_size")
            hyperparams["gamma"] = scheduler_params.get("gamma")

        elif scheduler == "ExponentialLR":
            hyperparams["gamma"] = scheduler_params.get("gamma")

        elif scheduler == "CyclicLR":
            # Typically, "base_lr" == the main lr
            hyperparams["base_lr"] = lr
            hyperparams["max_lr"] = scheduler_params.get("max_lr")
            hyperparams["step_size_up"] = scheduler_params.get("step_size_up")
            hyperparams["mode"] = scheduler_params.get("mode")

    return hyperparams


def compile_hyperparams_from_trial(trial):
    """
    Compiles all the hyperparameters suggested by a trial into a single dictionary.
    """
    hyperparams = {
        "batch_size": trial.params["batch_size"],
        "pos_weight_ratio": trial.params["pos_weight_ratio"],
        "num_layers": trial.params["num_layers"],
        "hidden_dim": trial.params["hidden_dim"],
        "heads": trial.params["heads"],
        "dropout": trial.params["dropout"],
        "node_embedding_type": trial.params["node_embedding"],
        "optimizer": trial.params["optimizer"],
        "lr": trial.params["lr"],
        "weight_decay": trial.params["weight_decay"],
        "scheduler": trial.params["scheduler"],
    }

    # Add RMSProp-specific params if the chosen optimizer is RMSProp
    if hyperparams["optimizer"] == "RMSprop":
        hyperparams["rmsprop_momentum"] = trial.params["rmsprop_momentum"]
        hyperparams["rmsprop_alpha"] = trial.params["rmsprop_alpha"]

    # Add scheduler-specific params if the chosen scheduler is not 'none'
    if hyperparams["scheduler"] != "none":
        if hyperparams["scheduler"] == "CosineAnnealingLR":
            hyperparams["T_max"] = trial.params["T_max"]
            hyperparams["eta_min"] = trial.params["eta_min"]
        elif hyperparams["scheduler"] == "ReduceLROnPlateau":
            hyperparams["factor"] = trial.params["reduce_factor"]
            hyperparams["patience"] = trial.params["reduce_patience"]
        elif hyperparams["scheduler"] == "StepLR":
            hyperparams["step_size"] = trial.params["step_size"]
            hyperparams["gamma"] = trial.params["step_gamma"]
        elif hyperparams["scheduler"] == "ExponentialLR":
            hyperparams["gamma"] = trial.params["exp_gamma"]
        elif hyperparams["scheduler"] == "CyclicLR":
            hyperparams["base_lr"] = trial.params["lr"]  # same as the main LR
            hyperparams["max_lr"] = trial.params["cyclic_max_lr"]
            hyperparams["step_size_up"] = trial.params["cyclic_step_size_up"]
            hyperparams["mode"] = trial.params["cyclic_mode"]

    return hyperparams



@ensure_wandb_init
def log_pruned(epoch, f1_score, trial):
    """
    Logs information about a pruned Optuna trial.

    Args:
        epoch (int): The epoch number at which the trial was pruned.
        f1_score (float): The F1 score at the time of pruning.
        trial (optuna.trial.Trial): The Optuna trial that was pruned.
    """
    wandb.log({"trial_pruned": trial.number, "epoch": epoch, "f1_score": f1_score})
    wandb.run.alert(
        title="Trial Pruned",
        text=f"Optuna pruned trial {trial.number}"
    )


@ensure_wandb_init
def log_nan_loss(epoch, trial=None):
    """
    Logs information about a trial that was pruned due to NaN loss.

    Args:
        epoch (int): The epoch number at which the trial was pruned.
        trial (optuna.trial.Trial): The Optuna trial that was pruned.
    """
    if trial:
        wandb.log({"trial_pruned": trial.number, "epoch": epoch, "is_nan": True})
    else:
        wandb.log({"epoch":epoch, "is_nan": True})
    wandb.finish(exit_code=-1)  # to mark the run as failed in wandb

@ensure_wandb_init
def log_test_metrics(f1_score, precision, recall, trial):
    """
    Logs test metrics after evaluating the model on the test set.

    Args:
        f1_score (float): F1 score on the test set.
        precision (float): Precision on the test set.
        recall (float): Recall on the test set.
        trial (optuna.trial.Trial): The current Optuna trial.
    """
    wandb.log({
        "meta/trial_number": trial.number,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1_score,
    })

@ensure_wandb_init
def log_evaluation_metrics(overall_precision, overall_recall, overall_accuracy, f1_score, epoch):
    """
    Logs evaluation metrics for validation.

    Args:
        overall_precision (float): Precision across all graphs.
        overall_recall (float): Recall across all graphs.
        overall_accuracy (float): Accuracy across all graphs.
        f1_score (float): F1 score across all graphs.
        epoch: Current epoch number.
    """
    wandb.log({
        "val/precision": overall_precision,
        "val/recall": overall_recall,
        "val/accuracy": overall_accuracy,
        "val/f1": f1_score,
        "epoch": epoch
    }, commit=False)  # Combine with epoch metrics

@ensure_wandb_init
def log_epoch_metrics(optimizer, total_loss, model, epoch):
    """
    Logs epoch-level metrics during training.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        total_loss (float): Total loss across the epoch.
        model (torch.nn.Module): Model used in training.
        epoch: Current epoch number.
    """
    weight_stats = {"zero": 0, "nan": 0, "inf": 0}
    grad_stats = {"zero": 0, "nan": 0, "inf": 0}
    weight_norm = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Weight statistics
                weight_norm += param.norm().item()

                weight_stats["zero"] += (param == 0).sum().item()
                weight_stats["nan"] += torch.isnan(param).sum().item()
                weight_stats["inf"] += torch.isinf(param).sum().item()

                # Gradient statistics
                if param.grad is not None:
                    grad = param.grad.data
                    grad_stats["zero"] += (grad == 0).sum().item()
                    grad_stats["nan"] += torch.isnan(grad).sum().item()
                    grad_stats["inf"] += torch.isinf(grad).sum().item()

    # Log all metrics in one call
    wandb.log({
        "train/loss": total_loss,
        "train/learning_rate": optimizer.param_groups[0]['lr'],
        "weights/norm": weight_norm,
        "weights/zero": weight_stats["zero"],
        "weights/nan": weight_stats["nan"],
        "weights/inf": weight_stats["inf"],
        "gradients/zero": grad_stats["zero"],
        "gradients/nan": grad_stats["nan"],
        "gradients/inf": grad_stats["inf"],
        "epoch": epoch
    })
