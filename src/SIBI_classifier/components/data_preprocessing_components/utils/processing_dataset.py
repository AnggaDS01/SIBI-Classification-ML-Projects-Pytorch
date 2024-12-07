import os
import torch
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(
            self, 
            file_paths: list, 
            label_index: int
        ) -> None:

        self.file_paths = file_paths
        self.label_index = label_index

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(
            self, 
            idx: int
        ) -> tuple:
        """
        Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the file path and the label. The file path is the path to the image file, and the
            label is the ground truth label for the image.
        """
        # Split the file path into its individual components
        split_text = self.file_paths[idx].split(os.path.sep)
        
        # The label is the ground truth label for the image. It is assumed to be located at the specified index in the
        # split text.
        label = split_text[self.label_index]
        
        # Return a tuple containing the file path and the label
        return self.file_paths[idx], label


class ConvertPathsToTensor(Dataset):
	def __init__(
			self, 
			datasets: list=None, 
			label_list: list=None, 
			seed: int=None, 
			transform: transforms=None
		):
		
		self.datasets = datasets
		self.transform = transform
		self.label_list = label_list
		self.seed = seed

	def __len__(self) -> int:
		return len(self.datasets)

	def __getitem__(self, idx) -> tuple:
		"""
		Retrieves a single item from the dataset.

		Args:
			idx (int): The index of the item to retrieve.

		Returns:
			tuple: A tuple containing the image tensor and the label tensor. The image tensor is a tensor containing the
			image data, and the label tensor is a tensor containing the ground truth label for the image.
		"""
		# Retrieve the file path and label for the given index
		file_path, label = self.datasets[idx]

		# Read the image at the given file path
		image = torchvision.io.read_image(file_path)

		# Convert the label to an index in the label list
		label_to_index = torch.tensor(self.label_list.index(label))

		# If a transform is specified, apply it to the image
		if self.transform is not None:
			# Set the seed for the transform so that it is reproducible
			torch.manual_seed(self.seed)
			# Apply the transform to the image
			image = self.transform(image)

		# Convert the image tensor to a float tensor
		image = image.to(torch.float32)
		# Convert the label tensor to a long tensor
		label_to_index = label_to_index.to(torch.long)
		# Return the image tensor and the label tensor as a tuple
		return image, label_to_index
