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
          hidden_size: number of feature dimension (input).
          num_labels: number of classes (output).
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

        self.learner = nn.Sequential(*all_layers)


    def forward(self, features, targets=None):
        """
        The forward step for the learner.
        """
        logits = self.learner(features)
        outputs = (logits, )

        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            outputs = (loss, ) + outputs

        return outputs

class AdversaryNN(nn.Module):
    def __init__(self, hidden_size, num_labels, n_hidden):
        """
        Implements the adversary DNN.
        Args:
          hidden_size: number of feature dimension (input).
          num_labels: number of classes (output).
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.weights = None

        all_layers = []
        input_size = hidden_size

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], num_labels))

        self.adversary = nn.Sequential(*all_layers)

    def forward(self, features):
        """
        The forward step for the adversary.
        """
        logits = self.adversary(features)
        weights = self.compute_example_weights(logits)
        self.weights = weights

        return weights

    def compute_example_weights(self, logits):
        if self.num_labels == 1:
            # Doing regression
            example_weights = torch.sigmoid(logits)
            mean_example_weights = example_weights.mean()
            example_weights = example_weights/torch.max(mean_example_weights, torch.tensor(1e-4))
            example_weights = torch.ones_like(example_weights) + example_weights

            return example_weights
        else:
            example_weights = torch.softmax(logits, dim=1)  
            mean_example_weights = example_weights.mean(dim=0)  
            example_weights = example_weights/torch.max(mean_example_weights, torch.tensor(1e-4)) 
            example_weights = torch.ones_like(example_weights) + example_weights 
            class_weights = example_weights[torch.arange(example_weights.size(0)), self.num_labels] 

            return class_weights

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
          hidden_size: number of feature dimension (input).
          num_labels: number of classes (output).
          learner_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          adversary_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          activation_fn: the activation function to use for the learner.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        # Mode to train the learner only, used outside of this module.
        self.pretrain = True

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
        if targets is None:
                learner_logits, = self.learner(features)
        else:
                learner_loss_raw, learner_logits = self.learner(features, targets)
        
        outputs = (learner_logits,)

        if targets is not None:
            batch_size = features.size(0)
            device = learner_loss_raw.device
            adversary_weights = torch.ones(batch_size, self.num_labels) if self.pretrain else self.adversary(features)
            adversary_weights = adversary_weights.to(device)
            loss = self.get_loss(learner_loss_raw, adversary_weights)
            outputs = loss + outputs

        return outputs

    def get_loss(self, learner_loss_raw, adversary_weights):
        """
        Wrapper for computing the learner loss and adversary loss.
        """
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

    def get_learner_named_parameters(self):
        return self.learner.named_parameters()

    def get_adversary_named_parameters(self):
        return self.adversary.named_parameters()

    def set_pretrain(self, value: bool):
        self.pretrain = value
