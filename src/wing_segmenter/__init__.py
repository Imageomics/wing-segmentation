__version__ = "0.1.0"

def __getattr__(name):
    if name == 'Segmenter':
        from wing_segmenter.segmenter import Segmenter
        return Segmenter
    raise AttributeError(f"module {__name__} has no attribute {name}")
