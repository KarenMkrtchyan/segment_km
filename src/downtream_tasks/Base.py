"""
Base class for all deep learning segmentation algorithms.
"""
from abc import ABC, abstractmethod
from scipy import ndimage as ndi

class BaseAnaliser(ABC):
    """Base class that all deep learning segmentation algorithms should inherit from."""
    
    def __init__(self, config=None):
        """
        Initialize the segmenter.
        
        Args:
            config (dict, optional): Configuration parameters for the segmenter.
        """
        self.config = config or {}
        
    @abstractmethod
    def prediction(self, images):
        """
        Segment the input images.
        
        Args:
            List of images (numpy.ndarray with shape NUM IMAGES * HEIGHT * WIDTH * 3): Input images to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        pass
    
    def probability(self, images_dir):
        """
        Preprocess the input image before segmentation.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        pass
    
