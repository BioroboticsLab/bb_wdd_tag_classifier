from typing import Literal, Tuple, Union, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
class TaggedBeeClassificationModel(nn.Module):
    def __init__(self):
        super(TaggedBeeClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 2 * 2, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @overload
    @staticmethod
    def postprocess_predictions(
        outputs: torch.Tensor, numpy: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @overload
    @staticmethod
    def postprocess_predictions(
        outputs: torch.Tensor, numpy: Literal[False]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    @staticmethod
    def postprocess_predictions(
        outputs: torch.Tensor, numpy: bool = True
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
    ]: ...

    @staticmethod
    def postprocess_predictions(outputs: torch.Tensor, numpy: bool = True):
        probabilities = F.softmax(outputs, 1)
        predictions = torch.argmax(outputs, 1)
        confidences = probabilities[np.arange(probabilities.shape[0]), predictions]
        tagged_probabilities = probabilities[np.arange(probabilities.shape[0]), 0]

        if numpy:
            predictions = predictions.detach().cpu().numpy()
            confidences = confidences.detach().cpu().numpy()
            tagged_probabilities = tagged_probabilities.detach().cpu().numpy()

        return predictions, confidences, tagged_probabilities
