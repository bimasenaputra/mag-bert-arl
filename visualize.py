import argparse
from typing import Any
from bert import MAG_BertForSequenceClassification, MAG_BertWithARL

parser = argparse.ArgumentParser()
parser.add_argument("--beta_shift", type=float, default=1.0, help="The constant 'beta' to be used in the adaption gate during feature fusion with other features")
parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of a neuron being dropped out during each training session")
parser.add_argument("--model", type=str, default="bert-base-uncased", help="Name of model to train")
parser.add_argument("--model_type", type=str, choices=["mag-bert-arl", "mag-bert"], default="mag-bert", help="MAG-BERT model type")
parser.add_argument("--num_labels", type=int, default=1, help="Number of classes/labels to predict")
args = parser.parse_args()

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

multimodal_config = MultimodalConfig(
    beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
)
if args.model_type == "mag-bert":
    model = MAG_BertForSequenceClassification.from_pretrained(args.model, multimodal_config=multimodal_config, num_labels=args.num_labels)
elif args.model_type == "mag-bert-arl":
    model = MAG_BertWithARL.from_pretrained(args.model, multimodal_config=multimodal_config, num_labels=args.num_labels)

def visualize_children(
    object : Any,
    level : int = 0,
) -> None:
    """
    Prints the children of (object) and their children too, if there are any.
    Uses the current depth (level) to print things in a ordonnate manner.
    """
    print(f"{'   ' * level}{level}- {type(object).__name__}")
    try:
        for child in object.children():
            visualize_children(child, level + 1)
    except:
        pass

visualize_children(model)
