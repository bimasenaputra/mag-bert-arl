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
        self, hidden_size, num_labels, n_hidden, activation_fn=nn.ReLU
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
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        all_layers = []
        input_size = hidden_size

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            all_layers.append(activation_fn())
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], num_labels))

        self.layers = nn.Sequential(*all_layers)


    def forward(self, features, targets=None):
        """
        The forward step for the learner.
        """

        # Get the logits output (for calculating loss)
        logits = self.layers(features)
        outputs = (logits, )

        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            outputs = outputs + (loss, )

        return outputs


class AdversaryNN(nn.Module):
    def __init__(self, hidden_size, num_labels, n_hidden):
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
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.weights = None

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
        weights = self.compute_example_weights(logits)
        self.weights = weights

        return weights

    def compute_example_weights(self, logits):
        if self.num_labels == 1:
            # Doing regression
            example_weights = torch.sigmoid(logits)
            mean_example_weights = example_weights.mean()
            example_weights /= torch.max(mean_example_weights, torch.tensor(1e-4))
            example_weights = torch.ones_like(example_weights) + example_weights

            return example_weights
        else:
            example_weights = torch.softmax(logits, dim=1)  
            mean_example_weights = example_weights.mean(dim=0)  
            example_weights /= torch.max(mean_example_weights, torch.tensor(1e-4)) 
            example_weights = torch.ones_like(example_weights) + example_weights 
            class_weights = example_weights[torch.arange(example_weights.size(0)), labels] 

            return class_weights

    def get_weights(self):
        return self.weights


class ARL(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_labels,
        learner_hidden_units=[64, 32],
        adversary_hidden_units=[32],
        activation_fn=nn.ReLU,
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
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pretrain = False

        self.learner = LearnerNN(
            hidden_size,
            num_labels,
            learner_hidden_units,
            activation_fn=activation_fn
        )
        self.adversary = AdversaryNN(
            hidden_size, num_labels, adversary_hidden_units
        )

    def forward(self, features, targets=None):
        """
        The forward step for the ARL.
        """
        learner_logits, learner_loss_raw = self.learner(features, targets)
        adversary_weights = self.adversary(features)

        outputs = (learner_logits,)

        if targets is not None:
            loss = self.get_loss(learner_loss_raw, features)
            outputs = loss + outputs

        return outputs

    def get_loss(self, learner_loss_raw, features):
        if self.pretrain:
            batch_size = features.size(0)
            adversary_weights = torch.ones(batch_size, self.num_labels)
        else:
            adversary_weights = self.adversary.get_weights()
        
        learner_loss = self.get_learner_loss(learner_loss_raw, adversary_weights)
        adversary_loss = self.get_adversary_loss(learner_loss_raw, adversary_weights)

        outputs = (learner_loss, adversary_loss,)

        return outputs

    def get_learner_loss(self, learner_loss_raw, adversary_weights):
        """
        Compute the loss for the learner.
        """
        weighted_loss = learner_loss_raw * adversary_weights
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, learner_loss_raw, adversary_weights):
        """
        Compute the loss for the adversary.
        """
        weighted_loss = -(adversary_weights * learner_loss_raw)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_learner_parameters(self):
        return self.learner.parameters()

    def get_adversary_parameters(self):
        return self.adversary.parameters()

    def set_pretrain(self, value):
        self.pretrain = value