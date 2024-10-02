class ModelLoadError(Exception):
    """Exception raised when a model fails to load."""
    pass

class ImageProcessingError(Exception):
    """Exception raised when an error occurs during image processing."""
    pass

class MetadataError(Exception):
    """Exception raised when there is an error with metadata handling."""
    pass
