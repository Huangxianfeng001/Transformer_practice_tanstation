import numpy as np
import torch
import matplotlib.pyplot as plt
from Transformer_components.Feedforward import FeedForward
print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
