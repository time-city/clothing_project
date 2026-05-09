"""
ai_model package - ResNet50 clothing classifier
"""

from .model import ClothingClassifier, load_model, predict, predict_batch, CLASS_NAMES

__all__ = ['ClothingClassifier', 'load_model', 'predict', 'predict_batch', 'CLASS_NAMES']
