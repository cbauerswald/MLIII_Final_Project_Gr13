"""
VGG16 model file for the shared runner (runner_project.py).

"""
import torch.nn as nn
from torchvision import models


def get_vgg16(num_classes=2, freeze=True): #num calss 2 bc theres fracture and non fracture
    """[Method] ImageNet-pretrained VGG16 with a new num_classes output layer.

    Same logic as the VGG16 branch of transfer_learning.py from hw 2
      1. load torchvision's VGG16 with ImageNet weights
      2. if freeze: freeze the convolution stack (features) and the three dense layers (classifier)
      3. replace layer 6 of the classifier (4096 -> 1000) with a new 4096 -> num_classes layer
    The new layer is created after freezing, so it is the only trainable part while frozen.
    The runner trains it first (head only), then unfreezes everything for full fine-tuning.
    vGG16 has no batch-norm layers
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = False
    last = model.classifier[6]
    model.classifier[6] = nn.Linear(last.in_features, num_classes)
    return model
