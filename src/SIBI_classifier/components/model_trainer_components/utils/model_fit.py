import os
import torch
import wandb
import time
import numpy as np

from tqdm import tqdm
from pathlib import Path

from SIBI_classifier.components.model_trainer_components.utils.setup_device_usage import setup_device
from SIBI_classifier.components.model_trainer_components.utils.data_loader import get_dataloader
from SIBI_classifier.logger.logging import *
from SIBI_classifier.components.model_trainer_components.utils.preparing_callbacks import *
from SIBI_classifier.components.model_trainer_components.utils.wandb_logs import *

COLOR_TEXT = 'yellow'

def train_step(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> tuple:
    """
    Performs a single training step (epoch) over the training dataset.

    Args:
        model (torch.nn.Module): The neural network model to train.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function used to evaluate the model.
        optim (torch.optim.Optimizer): Optimizer used to update the model parameters.

    Returns:
        tuple: A tuple containing the average training loss and accuracy for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    # Initialize variables to accumulate loss and accuracy over the entire training set
    train_loss, train_acc, total_train = 0.0, 0.0, 0

    # Iterate over batches of data from the training data loader
    for inputs, labels in tqdm(train_loader, total=(len(train_loader)), desc="Training"):
        # Move the input data and labels to the specified device (CPU or GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset the gradients of the model parameters
        optim.zero_grad()
        
        # Perform a forward pass through the model to get predictions
        outputs = model(inputs)

        # Calculate the loss using the criterion (e.g., cross-entropy loss)
        loss = criterion(outputs, labels)
        
        # Backpropagate the loss to compute gradients
        loss.backward()
        
        # Update the model parameters using the optimizer
        optim.step()

        # Accumulate the loss for this batch
        train_loss += loss.item()
        
        # Get the predicted class by finding the index with the maximum score
        _, predicted = torch.max(outputs, 1)
        
        # Accumulate the number of correct predictions
        train_acc += (predicted == labels).sum().item()
        
        # Accumulate the total number of samples processed
        total_train += labels.size(0)

    # Calculate the average training loss and accuracy over all batches
    avg_train_loss = train_loss / len(train_loader) / 100
    avg_train_acc = train_acc / total_train

    # Return the average loss and accuracy as a tuple
    return avg_train_loss, avg_train_acc

def valid_step(
    model: torch.nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    class_names: list,
    epoch: int
) -> tuple:
    """
    Performs a validation step over the validation dataset.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used to evaluate the model.
        class_names (list): List of class names for logging and reporting.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the average validation loss and accuracy for the epoch.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize variables to accumulate validation loss, accuracy, and total samples
    val_loss, val_acc, total_val = 0.0, 0.0, 0
    
    # Lists to store all true labels and predictions for further analysis
    all_labels, all_predictions = [], []
    
    # Flag to log a single batch of sample predictions to Weights & Biases
    save_sample_to_wandb = True

    # Evaluate model without tracking gradients to reduce memory usage
    with torch.inference_mode():
        # Iterate over batches of data from the validation data loader
        for inputs, labels in tqdm(val_loader, total=(len(val_loader)), desc="Validating"):
            # Move input data and labels to the specified device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Perform a forward pass through the model to get predictions
            outputs = model(inputs)
            
            # Calculate the loss using the criterion
            loss = criterion(outputs, labels)

            # Accumulate the loss for this batch
            val_loss += loss.item()

            # Get the predicted class by finding the index with the maximum score
            _, predicted = torch.max(outputs, 1)
            
            # Accumulate the number of correct predictions
            val_acc += (predicted == labels).sum().item()
            
            # Accumulate the total number of samples processed
            total_val += labels.size(0)

            # Append the current batch's labels and predictions to the lists
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

            # Log a sample of predictions to Weights & Biases once per epoch
            if save_sample_to_wandb:
                log_sample_predictions(inputs, labels, outputs, class_names, epoch, sample_count=5)
                save_sample_to_wandb = False

    # Calculate the average validation loss and accuracy over all batches
    avg_val_loss = (val_loss / len(val_loader)) / 100
    avg_val_acc = val_acc / total_val

    # Concatenate all labels and predictions for further analysis
    all_labels = np.concatenate(all_labels)
    all_predictions = np.vstack(all_predictions)
    
    # Plot and log the ROC curve to Weights & Biases
    roc_plot = plot_roc_curve(all_labels, all_predictions, class_names)
    wandb.log({"ROC Curve": wandb.Image(roc_plot)}, step=epoch)

    # Determine the predicted class for each sample
    predictions = np.argmax(all_predictions, axis=1)
    
    # Generate and log the confusion matrix to Weights & Biases
    cf_plot = log_confusion_matrix(all_labels, predictions, class_names)
    wandb.log({"Confusion Matrix": wandb.Image(cf_plot)}, step=epoch)

    # Generate and log the classification report as an image to Weights & Biases
    classification_report_image = log_classification_report_as_image(
        labels=all_labels,
        predictions=predictions,
        class_names=class_names
    )
    wandb.log({f"Classification Report": wandb.Image(classification_report_image)}, step=epoch)

    # Return the average validation loss and accuracy as a tuple
    return avg_val_loss, avg_val_acc

def fit(
    model: torch.nn.Module = None,
    train_pt_datasets: torch.utils.data.Dataset = None,
    valid_pt_datasets: torch.utils.data.Dataset = None,
    class_names: list = None,
    class_weights: list = None,
    epochs: int = 1,
    batch_size: int = 32,
    criterion: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    learning_rate: float = 1e-3,
    device_mode: str = 'cpu',
    num_gpus: int = 0,
    num_workers: int = int(os.cpu_count() / 2),
    model_file_path: Path = None,
):
    """
    Trains a neural network model over a specified number of epochs using the provided datasets.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_pt_datasets (Dataset): The training dataset.
        valid_pt_datasets (Dataset): The validation dataset.
        class_names (list): List of class names for logging purposes.
        class_weights (list): List of class weights for weighted loss calculation.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch to load.
        criterion (torch.nn.Module): Loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model parameters.
        learning_rate (float): Learning rate for the optimizer.
        device_mode (str): Mode indicating CPU or GPU usage.
        num_gpus (int): Number of GPUs to use for training.
        num_workers (int): Number of subprocesses to use for data loading.
        model_file_path (Path): Path for saving the model checkpoints.

    Returns:
        list: A list containing the history of training and validation metrics per epoch.
    """

    # Initialize the best validation loss to infinity
    best_val_loss = float('inf')

    # Initialize lists to store the training and validation loss and accuracy
    train_loss_list, valid_loss_list = [], []
    train_acc_list, valid_acc_list = [], []

    # Initialize a list to store the epoch history
    history = []

    # Setup the device to use
    model, device = setup_device(model, device_mode, num_gpus)

    # Create an optimizer object
    optimizer_obj = optimizer(model.parameters(), lr=learning_rate)

    # Create a criterion object
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion_obj = criterion(weight=class_weights_tensor)

    # Create a learning rate scheduler
    lr_scheduler = lr_scheduler_callback(optimizer_obj, step_size=10)

    # Create data loaders for the training and validation datasets
    train_loader, val_loader = get_dataloader(train_pt_datasets, valid_pt_datasets, batch_size, device_mode, num_workers)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        MODEL_TRAINING_LOGGER.info(f"Epoch {log_manager.color_text(epoch + 1, COLOR_TEXT)} of {log_manager.color_text(epochs, COLOR_TEXT)}")

        # Record the start time of the epoch
        start_time = time.time()

        # Perform a training step
        train_loss, train_acc = train_step(model, device, train_loader, criterion_obj, optimizer_obj)

        # Perform a validation step
        valid_loss, valid_acc = valid_step(model, device, val_loader, criterion_obj, class_names, epoch)

        # Record the time taken for the epoch
        epoch_time = time.time() - start_time

        # Log the epoch summary to Weights & Biases
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "epoch_time": epoch_time
            }
        )

        # Save the model checkpoint if the validation loss improves
        checkpoint_callback(model, model_file_path, epoch, valid_loss, best_val_loss)
        lr_scheduler.step()

        # Append the current epoch's results to the lists
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        # Append the current epoch's history to the history list
        history.append([epoch + 1, train_acc, valid_acc, train_loss, valid_loss, epoch_time])

        # Log the epoch summary
        MODEL_TRAINING_LOGGER.info(f"Epoch {log_manager.color_text(epoch + 1, COLOR_TEXT)} Summary:")
        MODEL_TRAINING_LOGGER.info(f"train_loss: {log_manager.color_text(train_loss, COLOR_TEXT)}, train_acc: {log_manager.color_text(train_acc, COLOR_TEXT)}")
        MODEL_TRAINING_LOGGER.info(f"valid_loss: {log_manager.color_text(f'{valid_loss:.4f}', COLOR_TEXT)}, valid_acc: {log_manager.color_text(f'{valid_acc:.4f}', COLOR_TEXT)}")
        MODEL_TRAINING_LOGGER.info('=' * 75)

    # Return the epoch history
    return history