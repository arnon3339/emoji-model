from modules import model
import tensorflow as tf
import os
import numpy as np
import random
import builtins
import toml
from modules import utils
from modules import tune

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

with open("config.toml", 'r') as f:
    builtins.CONFIG = toml.load(f)

if __name__ == "__main__":
    # utils.export_dataset()
    # model = model.AiModel()
    # model.fit()
    tune.run()