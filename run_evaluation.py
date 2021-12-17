import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from piano_transcription_inference.data_loader import (
    PFVNMix,
    PFVNPianoOnly,
    RevisedPFVNMix,
)
from piano_transcription_inference.evaluate import evaluate


def test_with_evaluate(dataset_path, eval_type, set_type):
    logdir = (
        Path("./runs") / f"exp_bytedance_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    )
    logdir.mkdir(exist_ok=True)
    group = [set_type]

    if eval_type == "mix":
        test_dataset = PFVNMix(path=dataset_path, groups=group)
    else:
        test_dataset = PFVNPianoOnly(path=dataset_path, groups=group)

    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)
        for batch in loader:
            track = Path(batch["path"][0]).parent.name
            batch_results = evaluate(batch, track)
            for key, value in batch_results.items():
                metrics[key].extend(value)
    print("")
    for key, value in metrics.items():
        if key[-2:] == "f1" or "loss" in key:
            print(f"{key} : {np.mean(value)}")

    with open(Path(logdir) / f"bytedance_results_{eval_type}_{set_type}.txt", "w") as f:
        for key, values in metrics.items():
            _, category, name = key.split("/")
            metric_string = f"{category:>32} {name:26}: "
            metric_string += f"{np.mean(values):.3f} +- {np.std(values):.3f}"
            print(metric_string)
            f.write(metric_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--eval-type", required=True)
    parser.add_argument("--set-type", default="test")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    eval_type = args.eval_type
    set_type = args.set_type

    test_with_evaluate(dataset_path, eval_type, set_type)
