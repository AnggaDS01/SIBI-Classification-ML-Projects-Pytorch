import torch 
from SIBI_classifier.logger.logging import *

COLOR_TEXT = 'yellow'

def model_checkpoint(model, model_file_path, epoch, valid_loss, best_val_loss):
    MODEL_TRAINING_LOGGER.info(f"Epoch {log_manager.color_text(epoch+1, COLOR_TEXT)} Summary:")
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), model_file_path)
        MODEL_TRAINING_LOGGER.info(f"Best validation loss: {log_manager.color_text(f'{best_val_loss:.4f}', COLOR_TEXT)} in {log_manager.color_text(model_file_path, COLOR_TEXT)}")
        MODEL_TRAINING_LOGGER.info(f"Saving best model for epoch: {log_manager.color_text(epoch+1, COLOR_TEXT)}")


def learning_rate_scheduler(optimizer, step_size, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, **kwargs)

    return lr_scheduler
