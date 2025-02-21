print("Downloading image datasets")

print("Post-processing images")

print("Predicting gaze fixations")

print("Analyzing results")

import os
from unisal.unisal.train import Trainer
from pathlib import Path

dataset_path = "./data"
verbose = True
trainer = Trainer.init_from_cfg_dir(Path('./unisal/training_runs/pretrained_unisal'))

for root, _, _ in os.walk(dataset_path):
    path = root.split(os.sep)
    if 'images' in path:
        if verbose:
            print(f"Processing dataset {os.sep.join(path[1:-1])}")
        trainer.generate_predictions_from_path(Path(root).parent, False, source='SALICON')