from SIBI_classifier.logger.logging import *
import torchvision.transforms as transforms 

COLOR_TEXT = 'yellow'

class ImageProcessor:
	def __init__(
			self, 
			image_size: tuple=(224, 224),
			mean=(0, 0, 0), 
			std=(1, 1, 1),
			brightness: tuple=(0.5, 1.5),
			contrast: tuple=(0.5, 1.5),
			saturation: tuple=(0.5, 1.5),
			hue: tuple=(-0.5, 0.5),
			p: float=0.5,
		):

		self.image_size = image_size
		self.mean = mean
		self.std = std
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue
		self.p = p
	
	def image_to_PIL(self) -> transforms.ToPILImage:
		IMAGE_PROCESSING_LOGGER.info(f"Normalizing image to {log_manager.color_text('0 to 1', COLOR_TEXT)}")
		return transforms.ToPILImage()

	def image_resizing(self) -> transforms.Resize:
		IMAGE_PROCESSING_LOGGER.info(f"Resizing image to {log_manager.color_text(self.image_size, COLOR_TEXT)}")
		return transforms.Resize(
			size=self.image_size[:2],
			interpolation=transforms.InterpolationMode.BILINEAR
		)

	def color_jitter(self) -> transforms.ColorJitter:
		IMAGE_PROCESSING_LOGGER.info(
			f"Color jittering with "
			f"brightness {log_manager.color_text(self.brightness, COLOR_TEXT)}, " 
			f"contrast {log_manager.color_text(self.contrast, COLOR_TEXT)}, " 
			f"saturation {log_manager.color_text(self.saturation, COLOR_TEXT)}, "
			f"hue {log_manager.color_text(self.hue, COLOR_TEXT)}"
		)
		return transforms.ColorJitter(
			brightness=self.brightness, 
			contrast=self.contrast, 
			saturation=self.saturation, 
			hue=self.hue
		)

	def hFlip(self) -> transforms.RandomHorizontalFlip:
		IMAGE_PROCESSING_LOGGER.info(f"Horizontal Flipping with probability {log_manager.color_text(self.p, COLOR_TEXT)}")
		return transforms.RandomHorizontalFlip(p=self.p)

	def vFlip(self) -> transforms.RandomVerticalFlip:
		IMAGE_PROCESSING_LOGGER.info(f"Vertical Flipping with probability {log_manager.color_text(self.p, COLOR_TEXT)}")
		return transforms.RandomVerticalFlip(p=self.p)
	
	def image_normalization(self) -> transforms.Normalize:
		IMAGE_PROCESSING_LOGGER.info(f"Normalizing image with mean {log_manager.color_text(self.mean, COLOR_TEXT)} and std {log_manager.color_text(self.std, COLOR_TEXT)}")
		return transforms.Normalize(mean=self.mean, std=self.std)

	def image_to_tensor(self) -> transforms.ToTensor:
		IMAGE_PROCESSING_LOGGER.info(f"Normalizing image to {log_manager.color_text('0 to 1', COLOR_TEXT)}")
		return transforms.ToTensor()