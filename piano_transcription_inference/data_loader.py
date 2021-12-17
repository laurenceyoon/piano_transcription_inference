import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import mido
import numpy as np
from pathlib import Path
import librosa

from abc import abstractmethod

import numpy as np
import pretty_midi

HOP_SIZE = 512
MAX_MIDI = 108
MIN_MIDI = 21
SAMPLE_RATE = 16000


# class PianoEventDataset(Dataset):
#     def __init__(self, config, path, group=None, debug=False):
#         super().__init__()
#         self.n_note = config.n_note
#         self.n_velocity = config.n_velocity
#         self.n_time = config.n_time
#         self.device = config.device

#         self.data = []
#         if group is None:
#             files = list(Path(path).glob("**"))
#         else:
#             files = list((Path(path) / group).glob("*"))

#         if debug:
#             files = files[:1000]

#         for fname in tqdm(list(files)):
#             self.data.append(torch.load(fname))

#         self.feat_ext = FeatureExtractor(config)

#     def __getitem__(self, idx):
#         audio = self.data[idx]["audio"].to(self.device)
#         feat = self.feat_ext(audio).squeeze().transpose(0, 1)  # (T, F)
#         feat_len = feat.size(0)
#         event = torch.zeros(
#             len(self.data[idx]["event"]) + 2,
#             self.n_note + self.n_velocity + self.n_time + 1,
#         ).to(
#             self.device
#         )  # (Sequence, Event)
#         event_len = event.size(0) - 1

#         for i, (note, velocity, time) in enumerate(self.data[idx]["event"]):
#             event[i + 1, note] = 1
#             event[i + 1, self.n_note + velocity] = 1
#             event[i + 1, self.n_note + self.n_velocity + round(time * 100)] = 1

#         event[-1, -1] = 1  # EOS

#         return feat, feat_len, event, event_len

#     def __len__(self):
#         return len(self.data)


class PianoSampleDataset(Dataset):
    def __init__(
        self, path, groups=None, sample_length=16000 * 5, hop_size=HOP_SIZE, seed=42,
    ):
        self.path = Path(path)
        self.groups = groups
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = None
        if sample_length is not None:
            self.sample_length = sample_length // hop_size * hop_size
        self.random = np.random.RandomState(seed)
        self.hop_size = hop_size

        self.file_list = dict()
        self.data = []

        print(f"Loading {len(groups)} group(s) of", self.__class__.__name__, "at", path)
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_files in tqdm(
                self.file_list[group], desc=f"Loading group {group}"
            ):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = {"path": data["path"]}

        audio = data["audio"]
        frames = data["frame"] >= 1
        onsets = data["onset"] >= 1
        velocity = data["velocity"]

        if self.sample_length is not None:
            n_steps = self.sample_length // self.hop_size

            step_begin = 0
            step_end = n_steps

            begin = step_begin * self.hop_size
            end = begin + self.sample_length

            audio_seg = audio[begin:end]
            frame_seg = frames[step_begin:step_end]
            onset_seg = onsets[step_begin:step_end]
            velocity_seg = velocity[step_begin:step_end, :]

            result["audio"] = audio_seg.float().div_(32768.0)
            result["frame"] = frame_seg.float()
            result["onset"] = onset_seg.float()
            result["velocity"] = velocity_seg.float().div_(128.0)
        else:
            result["audio"] = audio.float().div_(32768.0)
            result["frame"] = frames.float()
            result["onset"] = onsets.float()
            result["velocity"] = velocity.float().div_(128.0)
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """Returns the names of all available groups."""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """Returns the list of input files (audio_filename, tsv_filename) for this group."""
        raise NotImplementedError

    def load(self, audio_path, midi_path):
        """Loads an audio track and the corresponding labels."""
        # audio, sr = soundfile.read(audio_path, dtype="int16", samplerate=SAMPLE_RATE)
        audio, sr = librosa.load(audio_path, dtype="int16", sr=SAMPLE_RATE)
        assert sr == SAMPLE_RATE
        frames_per_sec = sr / self.hop_size

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        mel_length = audio_length // self.hop_size + 1

        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_length_sec = midi.get_end_time()
        frame_length = min(int(midi_length_sec * frames_per_sec), mel_length)

        audio = audio[: frame_length * self.hop_size]
        frame = midi.get_piano_roll(fs=frames_per_sec)
        onset = np.zeros_like(frame)
        velocity = np.zeros_like(frame)
        for inst in midi.instruments:
            for note in inst.notes:
                onset[note.pitch, int(note.start * frames_per_sec)] = 1
                velocity[note.pitch, int(note.start * frames_per_sec)] = note.velocity

        # to shape (time, pitch (88))
        frame = torch.from_numpy(frame[MIN_MIDI : MAX_MIDI + 1].T)
        onset = torch.from_numpy(onset[MIN_MIDI : MAX_MIDI + 1].T)
        velocity = torch.from_numpy(velocity[MIN_MIDI : MAX_MIDI + 1].T)
        return {
            "path": audio_path,
            "audio": audio,
            "frame": frame,
            "onset": onset,
            "velocity": velocity,
        }


class PFVNPianoOnly(PianoSampleDataset):
    def __init__(
        self, path, groups, sequence_length=None, hop_size=HOP_SIZE, seed=42,
    ):
        super().__init__(
            path, groups, sequence_length, hop_size, seed,
        )

    @classmethod
    def available_groups(cls):
        return ["test", "validation"]

    def get_track_range(self, group):
        if group == "validation":
            return range(30, 36)
        else:
            return range(36, 50)

    def files(self, group):
        track_range = self.get_track_range(group)
        return sorted(
            [
                ((track / "PF.wav").as_posix(), (track / "MIDI" / "PF.mid").as_posix(),)
                for track in self.path.iterdir()
                if int(track.name.split("Track")[1]) in track_range
            ]
        )


class PFVNMix(PianoSampleDataset):
    def __init__(
        self, path, groups, sequence_length=None, hop_size=HOP_SIZE, seed=42,
    ):
        super().__init__(
            path, groups, sequence_length, hop_size, seed,
        )

    @classmethod
    def available_groups(cls):
        return ["test", "validation"]

    def get_track_range(self, group):
        if group == "validation":
            return range(30, 36)
        else:
            return range(36, 50)

    def files(self, group):
        track_range = self.get_track_range(group)
        return sorted(
            [
                (
                    (track / "mix.wav").as_posix(),
                    (track / "MIDI" / "mix.mid").as_posix(),
                )
                for track in self.path.iterdir()
                if int(track.name.split("Track")[1]) in track_range
            ]
        )


class RevisedPFVNMix(Dataset):
    def __init__(
        self, path, groups=None, sample_length=16000 * 5, hop_size=HOP_SIZE, seed=42,
    ):
        self.path = Path(path)
        self.groups = groups
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = None
        if sample_length is not None:
            self.sample_length = sample_length // hop_size * hop_size
        self.random = np.random.RandomState(seed)
        self.hop_size = hop_size

        self.file_list = dict()
        self.data = []

        print(f"Loading {len(groups)} group(s) of", self.__class__.__name__, "at", path)
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_files in tqdm(
                self.file_list[group], desc=f"Loading group {group}"
            ):
                self.data.append(self.load(input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = {"path": data["path"]}

        frames = data["frame"] >= 1
        onsets = data["onset"] >= 1
        velocity = data["velocity"]

        if self.sample_length is not None:
            n_steps = self.sample_length // self.hop_size

            step_begin = 0
            step_end = n_steps

            frame_seg = frames[step_begin:step_end]
            onset_seg = onsets[step_begin:step_end]
            velocity_seg = velocity[step_begin:step_end, :]

            result["frame"] = frame_seg.float()
            result["onset"] = onset_seg.float()
            result["velocity"] = velocity_seg.float().div_(128.0)
        else:

            result["frame"] = frames.float()
            result["onset"] = onsets.float()
            result["velocity"] = velocity.float().div_(128.0)
        return result

    def __len__(self):
        return len(self.data)

    def get_track_range(self, group):
        if group == "validation":
            return range(30, 36)
        else:
            return range(36, 50)

    @classmethod
    def available_groups(cls):
        """Returns the names of all available groups."""
        return ["test", "validation"]

    def files(self, group):
        """Returns the list of input files (audio_filename, tsv_filename) for this group."""
        track_range = self.get_track_range(group)
        return sorted(
            [
                (track / "MIDI" / "mix.mid").as_posix()
                for track in self.path.iterdir()
                if int(track.name.split("Track")[1]) in track_range
            ]
        )

    def load(self, midi_path):
        """Loads an audio track and the corresponding labels."""
        assert Path(midi_path).exists()

        frames_per_sec = SAMPLE_RATE / HOP_SIZE
        midi = pretty_midi.PrettyMIDI(midi_path)

        frames, onsets, velocities = dict(), dict(), dict()
        for inst in midi.instruments:
            frame = inst.get_piano_roll(fs=frames_per_sec)
            onset = np.zeros_like(frame)
            velocity = np.zeros_like(frame)
            for note in inst.notes:
                onset[note.pitch, int(note.start * frames_per_sec)] = 1
                velocity[
                    note.pitch,
                    int(note.start * frames_per_sec) : int(note.end * frames_per_sec),
                ] = note.velocity

            frame = torch.from_numpy(frame[MIN_MIDI : MAX_MIDI + 1].T)
            onset = torch.from_numpy(onset[MIN_MIDI : MAX_MIDI + 1].T)
            velocity = torch.from_numpy(velocity[MIN_MIDI : MAX_MIDI + 1].T)

            frames[inst.name] = frame
            onsets[inst.name] = onset
            velocities[inst.name] = velocity

        return {
            "path": midi_path,
            "frame": frames,
            "onset": onsets,
            "velocity": velocities,
        }
