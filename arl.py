###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains an implementation of the ARL model prented in "Fairness 
# without Demographics through Adversarially Reweighted Learning" by Lahoti 
# et al..
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnerNN(nn.Module):
    def __init__(
        self, hidden_size, num_labels, n_hidden, 
        activation_fn=nn.ReLU, device='cpu'
    ):
        """
        Implements the learner DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                           embedding sizes for categorical columns.
          num_labels: number of classes.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
          activation_fn: the activation function to use.
        """
        super().__init__()
        self.device = device

        all_layers = []
        input_size = hidden_size

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            all_layers.append(activation_fn())
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], num_labels))

        self.layers = nn.Sequential(*all_layers)


    def forward(self, features, labels=None):
        """
        The forward step for the learner.
        """

        # Get the logits output (for calculating loss)
        logits = self.layers(features)
        logits.to(self.device)

        return logits


class AdversaryNN(nn.Module):
    def __init__(self, hidden_size, num_labels, n_hidden device='cpu'):
        """
        Implements the adversary DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                          embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
        """
        super().__init__()
        self.device = device

        all_layers = []
        input_size = hidden_size

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], num_labels))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, features):
        """
        The forward step for the adversary.
        """
        logits = self.layers(features)
        logits.to(self.device)

        return logits


class ARL(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_labels,
        learner_hidden_units=[64, 32],
        adversary_hidden_units=[32],
        batch_size=256,
        activation_fn=nn.ReLU,
        device='cpu',
    ):
        """
        Combines the Learner and Adversary into a single module.

        Args:
          embedding_size: list of tuples (n_classes, embedding_dim) containing
                    embedding sizes for categorical columns.
          n_num_cols: the amount of numerical columns in the data.
          learner_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          adversary_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          batch_size: the batch size.
          activation_fn: the activation function to use for the learner.
        """
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.adversary_weights = torch.ones(batch_size, 1)

        self.learner = LearnerNN(
            hidden_size,
            num_labels,
            learner_hidden_units,
            activation_fn=activation_fn,
            device=device
        )
        self.adversary = AdversaryNN(
            hidden_size, num_labels, adversary_hidden_units, device=device
        )

        self.learner.to(device)
        self.adversary.to(device)

    def forward(self, features, targets=None):
        """
        The forward step for the ARL.
        """
        learner_logits = self.learner(features)
        adversary_logits = self.adversary(features)

        outputs = (learner_logits,)

        if targets is not None:
            learner_loss = self.get_learner_loss(learner_logits, targets)
            adversary_loss = self.get_adversary_loss(learner_logits, targets, adversary_logits)

            outputs = (learner_loss, adversary_loss,) + outputs

        return outputs

    def get_learner_loss(self, logits, targets):
        """
        Compute the loss for the learner.
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.to(self.device)

        adversary_weights = self.adversary_weights.to(self.device)
        weighted_loss = loss * adversary_weights
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, logits, targets, adv_logits):
        """
        Compute the loss for the adversary.
        """
        adversary_weights = self.compute_example_weights(adv_logits)
        self.adversary_weights = adversary_weights.detach()
        logits = logits.detach()

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.to(self.device)

        weighted_loss = -(adversary_weights * loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def compute_example_weights(self, adv_output_layer):
        if self.num_labels == 1:
            # Doing regression
            example_weights = torch.sigmoid(adv_output_layer)
            mean_example_weights = example_weights.mean()
            example_weights /= torch.max(mean_example_weights, torch.tensor(1e-4))
            example_weights = torch.ones_like(example_weights) + example_weights

            return example_weights
        else:
            example_weights = torch.softmax(adv_output_layer, dim=1)  
            mean_example_weights = example_weights.mean(dim=0)  
            example_weights /= torch.max(mean_example_weights, torch.tensor(1e-4)) 
            example_weights = torch.ones_like(example_weights) + example_weights 
            class_weights = example_weights[torch.arange(example_weights.size(0)), labels] 

            return class_weights

    def learner_zero_grad(self):
        self.learner.zero_grad()

    def adversary_zero_grad(self):
        self.adversary.zero_grad()