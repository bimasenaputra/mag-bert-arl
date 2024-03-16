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

        for i in range(n_hidden):
            layer = []
            layer.append(nn.Linear(hidden_size, hidden_size))
            layer.append(nn.Dropout(0.5))
            all_layers.append(nn.Sequential(*layer))
            
            layer = []
            layer.append(nn.LayerNorm(hidden_size))
            layer.append(activation_fn())
            all_layers.append(nn.Sequential(*layer))

        all_layers.append(nn.Linear(hidden_size, num_labels))

        self.learner = nn.ModuleList(all_layers)

    def forward(self, features, targets=None):
        """
        The forward step for the learner.
        """
        res_features = features
        for i, layer in enumerate(self.learner):
            if i % 2 == 1:
                continue
            if i > 0:
                features = self.learner[i-1](features + res_features)
                res_features = features
            features =  self.learner[i](features)
        logits = features
        outputs = (logits, )

        if targets is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), targets.view(-1))
                #loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), targets.view(-1))
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

        all_layers = []

        for i in range(n_hidden):
            layer = []
            layer.append(nn.Linear(hidden_size, hidden_size))
            layer.append(nn.Dropout(0.5))
            all_layers.append(nn.Sequential(*layer))
            all_layers.append(nn.LayerNorm(hidden_size))

        all_layers.append(nn.Linear(hidden_size, num_labels))

        self.adversary = nn.ModuleList(all_layers)

    def forward(self, features):
        """
        The forward step for the adversary.
        """
        res_features = features
        for i, layer in enumerate(self.adversary):
            if i % 2 == 1:
                continue
            if i > 0:
                features = self.adversary[i-i](features + res_features)
                res_features = features
            features = self.adversary[i](features)
        logits = features
        weights = self.compute_example_weights(logits)
        return weights

    def compute_example_weights(self, logits):
        if self.num_labels == 1:
            # Doing regression
            example_weights = F.relu(logits)
            #example_weights = torch.sigmoid(logits)
            mean_example_weights = example_weights.mean()
            example_weights = example_weights/torch.max(mean_example_weights, torch.tensor(1e-4))
            example_weights = torch.ones_like(example_weights) + example_weights
            
            return example_weights
        elif self.num_labels == 2:
            # Doing binary classification
            example_weights = torch.sigmoid(logits, dim=1)
            mean_example_weights = example_weights.mean(dim=0)
            example_weights = example_weights/torch.max(mean_example_weights, torch.tensor(1e-4))
            example_weights = torch.ones_like(example_weights) + example_weights
            class_weights = example_weights[torch.arange(example_weights.size(0)), self.num_labels]
            return class_weights
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
        learner_hidden_units=3,
        adversary_hidden_units=2,
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
            print(learner_loss_raw)
            print(loss)
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
