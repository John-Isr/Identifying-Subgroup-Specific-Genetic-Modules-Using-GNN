import torch
import gc
from torch import nn
import optuna

from utils import load_data_splits, init_wandb, compile_hyperparams_from_trial
from models import GNNClassifier
from models import train_model, evaluate_model
from utils import log_test_metrics, finish_logging


def objective(trial):

    ## Getting optuna parameters
    # Training
    batch_size = trial.suggest_int("batch_size", 1, 32)

    # Loss Function
    pos_weight_ratio = trial.suggest_int("pos_weight_ratio", 10, 20)

    # Model Architecture
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    heads = 2 ** trial.suggest_int("heads", 0, 3) # This will suggest powers of 2, logged to wandb appropriately though
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    node_embedding_type = trial.suggest_categorical(
        "node_embedding", ["weighted", "spectral_positional_encoding", "unweighted"]
    )

    # Optimizer Configuration
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adamax", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    optimizer_params = {}
    if optimizer_name == "RMSprop":
        # RMSProp-specific parameters
        optimizer_params["momentum"] = trial.suggest_float("rmsprop_momentum", 0.8, 0.99)
        optimizer_params["alpha"] = trial.suggest_float("rmsprop_alpha", 0.9, 0.99)

    # Scheduler Configuration
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR",
                      "ReduceLROnPlateau", "CyclicLR", "none"]
    )
    scheduler_params = {}

    if scheduler_name != "none":
        if scheduler_name == "CosineAnnealingLR":
            scheduler_params["T_max"] = trial.suggest_int("T_max", 50, 200)
            scheduler_params["eta_min"] = trial.suggest_float("eta_min", 1e-6, 1e-4, log=True)

        elif scheduler_name == "ReduceLROnPlateau":
            # Parameters specific to ReduceLROnPlateau
            scheduler_params["factor"] = trial.suggest_float("reduce_factor", 0.1, 0.5)
            scheduler_params["patience"] = trial.suggest_int("reduce_patience", 3, 10)
            scheduler_params["mode"] = "max"  # Assuming F1 is the metric monitored

        elif scheduler_name == "StepLR":
            scheduler_params["step_size"] = trial.suggest_int("step_size", 20, 50)
            scheduler_params["gamma"] = trial.suggest_float("step_gamma", 0.1, 0.9)

        elif scheduler_name == "ExponentialLR":
            scheduler_params["gamma"] = trial.suggest_float("exp_gamma", 0.8, 0.99)

        elif scheduler_name == "CyclicLR":
            # CyclicLR parameters (base_lr = optimizer's lr, max_lr is tuned)
            scheduler_params["base_lr"] = lr  # Use the optimizer's initial lr
            scheduler_params["max_lr"] = trial.suggest_float("cyclic_max_lr", 1e-3, 1e-2, log=True)
            scheduler_params["step_size_up"] = trial.suggest_int("cyclic_step_size_up", 50, 200)
            scheduler_params["mode"] = trial.suggest_categorical("cyclic_mode", ["triangular", "triangular2"])
            # Dynamically set cycle_momentum
            scheduler_params["cycle_momentum"] = optimizer_name == "RMSprop"

    ## Initialize wandb using this run configuration
    config = compile_hyperparams_from_trial(trial)
    init_wandb(config)
    ## Initialize all the training and model components
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gc.collect()  # Collect garbage from previous trials

    # Get DataLoaders
    train_loader, val_loader,test_loader = load_data_splits(batch_size,node_embedding_type=node_embedding_type)
    # Loss Function
    pos_weight = torch.tensor([pos_weight_ratio], dtype=torch.float32).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # Build Model
    # Get a batch from the DataLoader
    batch = next(iter(train_loader))

    # Convert batch back to a list of individual graphs
    graphs = batch.to_data_list()  # Use PyG's built-in function

    # Access a specific graph from the batch
    graph = graphs[0]
    input_dim = graph.x.size(1)  # Number of node feature dimensions
    edge_dim = graph.edge_attr.size(1)  # Number of edge feature dimensions

    model = GNNClassifier(
        input_dim=input_dim,
        edge_dim=edge_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
    )
    model.to(device)
    # Initialize optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_params  # Passes empty dict for non-RMSProp optimizers
    )

    # Initialize scheduler
    if scheduler_name != "none":
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_class(optimizer, **scheduler_params)
    else:
        scheduler = None


    ## Execute Training, test and return
    trained_model = None
    try:
        trained_model = train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=250,  # Fixed
            trial=trial
        )
    except optuna.exceptions.TrialPruned as e:
        # Evaluate on test set before gracefully exiting
        test_metrics = evaluate_model(model, test_loader)
        log_test_metrics(test_metrics['f1_score'], test_metrics['precision'], test_metrics['recall'], trial)
        finish_logging()
        clean_up_trial(model, device)
        raise e
    except ValueError as e:
        # We can't assess on the test set if the model reached Nan/Inf loss
        clean_up_trial(model, device)
        raise e
    except RuntimeError as e:
        error_message = str(e).lower()  # Normalize to lowercase for case-insensitive matching
        if "out of memory" in error_message or "cuda" in error_message:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()  # Tell Optuna to discard this trial


    # Evaluate on test set before gracefully exiting
    test_metrics = evaluate_model(model, test_loader)
    log_test_metrics(test_metrics['f1_score'], test_metrics['precision'], test_metrics['recall'], trial)
    finish_logging()
    clean_up_trial(model, device)
    return test_metrics['f1_score']

def clean_up_trial(model, device):
    # Clean up trial resources
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
