import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from piano_transcription_inference.evaluate import evaluate


def get_track_range(group):
    if group == "validation":
        return range(30, 35)
    else:
        return range(35, 51)


def test_with_evaluate(dataset_path: Path, transcribed_path: Path, set_type):
    logdir = (
        Path("./runs") / f"exp_bytedance_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    )
    logdir.mkdir(exist_ok=True)
    track_range = get_track_range(set_type)

    track_group = sorted(
        [
            track / "MIDI" / "PF.mid"
            for track in dataset_path.iterdir()
            if int(track.name.split("Track")[1]) in track_range
        ]
    )
    print(track_group)
    metrics = defaultdict(list)
    for track in tqdm(track_group):
        results = evaluate(track, transcribed_path)
        for key, value in results.items():
            metrics[key].extend(value)

    print("")
    for key, value in metrics.items():
        if key[-2:] == "f1" or "loss" in key:
            print(f"{key} : {np.mean(value)}")

    with open(Path(logdir) / f"bytedance_results_{set_type}.txt", "w") as f:
        for key, values in metrics.items():
            _, category, name = key.split("/")
            metric_string = f"{category:>32} {name:26}: "
            metric_string += f"{np.mean(values):.3f} +- {np.std(values):.3f}"
            print(metric_string)
            f.write(metric_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--transcribed-path", required=True)
    parser.add_argument("--set-type", default="test")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    transcribed_path = Path(args.transcribed_path)
    set_type = args.set_type

    test_with_evaluate(dataset_path, transcribed_path, set_type)
