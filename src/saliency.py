import os
from pathlib import Path

def generate_saliency_maps(dataset_path, model_path, verbose=False): 
    trainer = Trainer.init_from_cfg_dir(Path(model_path))

    for root, _, files in os.walk(dataset_path):
        path = root.split(os.sep)
        if 'images' in path:
            if verbose:
                print(f"Processing dataset {os.sep.join(path[1:-1])}")

            trainer.generate_predictions_from_path(Path(root).parent, False, source='SALICON')

if __name__ == "__main__":
    generate_saliency_maps("../data", "../unisal/training_runs/pretrained_unisal", verbose=True)