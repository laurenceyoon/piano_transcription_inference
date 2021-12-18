import argparse
import os
from pathlib import Path


def transcribe(source_dir, target_dir, start, stop, transcribe_type):
    dir_range = range(start, stop) if start and stop else None
    source_dir = Path(source_dir)
    assert source_dir.exists()
    save_path = Path(target_dir)
    save_path.mkdir(exist_ok=True)

    for track in source_dir.iterdir():
        if not track.name.startswith("Track"):
            continue

        n_track = int(track.name.split("Track")[1])
        if dir_range and n_track not in dir_range:
            continue

        print(f"transcribing track: {track}")
        input_path = track / f"{transcribe_type}.wav"
        print(input_path)
        output_path = save_path / f"{track.name}.mid"

        command = f"python3 example.py --audio_path={input_path.as_posix()} --output_midi_path={output_path.as_posix()} --cuda"

        os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--start", required=False)
    parser.add_argument("--stop", required=False)
    parser.add_argument("--transcribe-type", required=True, help="mix or PF")
    args = parser.parse_args()

    start, stop = None, None
    if args.start:
        start = int(args.start)
        stop = int(args.stop)

    transcribe(args.source_path, args.save_path, start, stop, args.transcribe_type)
