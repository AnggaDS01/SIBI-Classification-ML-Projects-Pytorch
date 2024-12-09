import torch

# Functional API - Define the forward pass
class TransferLearningModel(torch.nn.Module):
	def __init__(
			self, 
			base_model: torch.nn.Module=None,
			class_names: list=None
		) -> None:
        
		super(TransferLearningModel, self).__init__()

		length_label_list = len(class_names)
		num_classes = 1 if length_label_list == 2 else length_label_list

		self.base_model = base_model
		self.base_model.features = base_model.features  # Feature extractor
		self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

		# Define a new fully cotorch.nnected layer
		self.fc = torch.nn.Linear(base_model.classifier.in_features, num_classes)

		# Activation based on number of classes
		self.activation = torch.nn.Sigmoid() if num_classes == 1 else torch.nn.Softmax(dim=1)

	def forward(
			self, 
			x: torch.Tensor
        ) -> torch.Tensor:
		
		# Pass through the feature extractor
		x = self.base_model.features(x)
		
		# Apply global average pooling
		x = self.global_avg_pool(x)
		x = torch.flatten(x, 1)  # Flatten the tensor
		
		# Pass through the classifier
		x = self.fc(x)

		# Apply activation
		x = self.activation(x)
		return x