import os, sys
import torch

from SIBI_classifier.logger.logging import *
from SIBI_classifier.exception import SIBIClassificationException

COLOR_TEXT = 'yellow'

# Setup model untuk fleksibilitas
def setup_device(
        model: torch.nn.Module=None, 
        device_mode: str='cpu', 
        num_gpus: int=0
    ) -> tuple:
		
	try:
		if device_mode == "multi-gpu" and num_gpus > 1:
			torch.distributed.init_process_group(backend='nccl')
			local_rank = int(os.environ["LOCAL_RANK"])  # Rank di multi-proses
			torch.cuda.set_device(local_rank)
			device = torch.device(f"cuda:{local_rank}")
			model = model.to(device)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
		elif device_mode == "single-gpu" and num_gpus == 1:
			device = torch.device("cuda:0")
			model = model.to(device)
		else:  # CPU
			device = torch.device("cpu")
			model = model.to(device)

		SETUP_MODEL_LOGGER.info(f"Is CUDA available: {log_manager.color_text(torch.cuda.is_available(), COLOR_TEXT)}")
		SETUP_MODEL_LOGGER.info(f"Number of GPUs: {log_manager.color_text((device_mode, num_gpus), COLOR_TEXT)}")
		SETUP_MODEL_LOGGER.info(f"CUDA Device Name: {log_manager.color_text(torch.cuda.get_device_name(0), COLOR_TEXT)}")
		SETUP_MODEL_LOGGER.info(f"Is model on CUDA: {log_manager.color_text(next(model.parameters()).is_cuda, COLOR_TEXT)}")

		return model, device

	except Exception as e:
		raise SIBIClassificationException(e, sys)
	
# Cek perangkat yang tersedia
def get_device():
	try:
		if torch.cuda.is_available():
			num_gpus = torch.cuda.device_count()
			if num_gpus > 1:
				device_mode = "multi-gpu"
			elif num_gpus == 1:
				device_mode = "single-gpu"
			else:
				device_mode = "cpu"

		SETUP_MODEL_LOGGER.info(f"Device mode: {log_manager.color_text(device_mode, COLOR_TEXT)}, Number of GPUs: {log_manager.color_text(num_gpus, COLOR_TEXT)}")
		return device_mode, num_gpus

	except Exception as e:
		raise SIBIClassificationException(e, sys)