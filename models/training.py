import math
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from utils import log_epoch_metrics, log_evaluation_metrics, log_pruned, log_nan_loss
import optuna


def train_model(model, optimizer, scheduler, loss_function, train_loader, val_loader, num_epochs, trial=None):
    """
    Trains the given model and evaluates it at each epoch.

    Args:
        model (torch.nn.Module): The GNN model to train, already on device.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loss_function (torch.nn.Module): The loss function.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of epochs to train for.
        trial (optuna.Trial, optional): Optuna trial for hyperparameter tuning.
    returns:
        model: Trained model
    """
    device = next(model.parameters()).device  # Ensure consistency with model device
    for epoch in range(num_epochs):
        model.train()
        total_loss = torch.zeros(1, device=device)  # GPU tensor to accumulate loss

        for batch_idx, batch in enumerate(train_loader):
            if device == 'cuda':
                batch = batch.to(device, non_blocking=True)
            else:
                batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)

            # Compute loss
            loss = loss_function(out.squeeze(), batch.y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step scheduler if needed
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                scheduler.step()

            total_loss += loss.detach()
        loss_for_logging = total_loss.item()

        # Handle NaN or Inf cases
        if math.isnan(loss_for_logging) or math.isinf(loss_for_logging):
            log_nan_loss(epoch, trial)
            raise ValueError(f"NaN or Inf loss at epoch {epoch}")


        # Evaluate the model
        val_metrics = evaluate_model(model, val_loader)
        # Log epoch and validation metrics
        log_evaluation_metrics(val_metrics['precision'], val_metrics['recall'], val_metrics['accuracy'],
                               val_metrics['f1_score'], epoch)
        log_epoch_metrics(optimizer, total_loss, model, epoch)

        if scheduler:
            if isinstance(scheduler, (torch.optim.lr_scheduler.StepLR,
                                     torch.optim.lr_scheduler.ExponentialLR,
                                     torch.optim.lr_scheduler.CosineAnnealingLR)):
                scheduler.step()
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # We want to reduce on F1 improvements,
                scheduler.step(val_metrics['f1_score'])

        # Optuna pruning check
        if trial:
            f1 = val_metrics['f1_score']
            trial.report(f1, epoch)

            if trial.should_prune():
                log_pruned(epoch, f1, trial)
                raise optuna.TrialPruned()
    return model


def evaluate_model(model, data_loader):
    """
    Evaluates the given model on the provided data loader.

    Args:
        model (torch.nn.Module): The GNN model to evaluate, already on device.
        data_loader (DataLoader): DataLoader for the evaluation set.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    device = next(model.parameters()).device  # Ensure consistency with model device
    model.eval()

    all_outs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)

            # Accumulate on GPU
            all_outs.append(torch.sigmoid(out))
            all_targets.append(batch.y)

    # Concatenate once at the end
    all_outs = torch.cat(all_outs, dim=0)  # shape [total_examples, ...]
    all_targets = torch.cat(all_targets, dim=0)

    # Now we do a single move to CPU
    all_outs_cpu = all_outs.cpu().numpy()
    all_targets_cpu = all_targets.cpu().numpy()

    # Threshold predictions at 0.5
    y_pred = (all_outs_cpu > 0.5)
    y_true = all_targets_cpu


    # Compute evaluation metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"precision": round(precision, 4), "recall": round(recall, 4), "accuracy": round(accuracy, 4), "f1_score": round(f1, 4)}

