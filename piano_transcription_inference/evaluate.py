from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval import transcription_velocity
from mir_eval.util import midi_to_hz

from piano_transcription_inference.config import HOP_SIZE, MIN_MIDI, SAMPLE_RATE
from piano_transcription_inference.utilities import load_frame_and_onset_from_midi

INST_GROUP = ["PF", "VN"]


def evaluate(batch, track_name):
    metrics = defaultdict(list)

    midi_path = Path("./PF-VN-validation-results") / f"{track_name}.mid"
    transcribed_data = load_frame_and_onset_from_midi(midi_path.as_posix())
    pred_frames = transcribed_data["frame"]
    pred_onsets = transcribed_data["onset"]
    pred_velocity = transcribed_data["velocity"]

    for batch_idx in range(batch["audio"].shape[0]):
        pred_frame = torch.sigmoid(pred_frames)
        pred_onset = torch.sigmoid(pred_onsets)
        pred_velocity = torch.sigmoid(pred_velocity)

        label_frame = batch["frame"][batch_idx]
        label_onset = batch["onset"][batch_idx]
        label_velocity = batch["velocity"][batch_idx]

        if pred_frame.shape != label_frame.shape:
            pred_frame = pred_frame[: label_frame.shape[0], :]
            pred_onset = pred_onset[: label_onset.shape[0], :]
            pred_velocity = pred_velocity[: label_velocity.shape[0], :]

        pr, re, f1 = framewise_eval(pred_frame, label_frame)
        metrics["metric/frame/frame_precision"].append(pr)
        metrics["metric/frame/frame_recall"].append(re)
        metrics["metric/frame/frame_f1"].append(f1)

        pr, re, f1 = framewise_eval(pred_onset, label_onset)
        metrics["metric/frame/onset_precision"].append(pr)
        metrics["metric/frame/onset_recall"].append(re)
        metrics["metric/frame/onset_f1"].append(f1)

        p_est, i_est, v_est = extract_notes(pred_onset, pred_frame, pred_velocity)
        p_ref, i_ref, v_ref = extract_notes(label_onset, label_frame, label_velocity)

        scaling = HOP_SIZE / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_est])

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics["metric/note/precision"].append(p)
        metrics["metric/note/recall"].append(r)
        metrics["metric/note/f1"].append(f)
        metrics["metric/note/overlap"].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics["metric/note-with-offsets/precision"].append(p)
        metrics["metric/note-with-offsets/recall"].append(r)
        metrics["metric/note-with-offsets/f1"].append(f)
        metrics["metric/note-with-offsets/overlap"].append(o)

        p, r, f, o = transcription_velocity.precision_recall_f1_overlap(
            i_ref,
            p_ref,
            v_ref,
            i_est,
            p_est,
            v_est,
            onset_tolerance=0.05,
            offset_min_tolerance=0.05,
            velocity_tolerance=0.1,
        )
        metrics["metric/note-with-offsets-velocity/precision"].append(p)
        metrics["metric/note-with-offsets-velocity/recall"].append(r)
        metrics["metric/note-with-offsets-velocity/f1"].append(f)
        metrics["metric/note-with-offsets-velocity/overlap"].append(o)

    return metrics


def framewise_eval(pred, label, threshold=0.5):
    """Evaluates frame-wise (point-wise) evaluation.

    Args:
        pred: torch.Tensor of shape (frame, pitch)
        label: torch.Tensor of shape (frame, pitch)
    """

    tp = torch.sum((pred >= threshold) * (label == 1)).cpu().numpy()
    fn = torch.sum((pred < threshold) * (label == 1)).cpu().numpy()
    fp = torch.sum((pred >= threshold) * (label != 1)).cpu().numpy()

    pr = tp / float(tp + fp) if (tp + fp) > 0 else 0
    re = tp / float(tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pr * re / float(pr + re) if (pr + re) > 0 else 0

    return pr, re, f1


# def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
#     """Finds the note timings based on the onsets and frames information.

#     Args:
#         onsets: torch.FloatTensor of shape (frames, bins)
#         frames: torch.FloatTensor of shape (frames, bins)
#         onset_threshold: float
#         frame_threshold: float

#     Returns:
#         pitches: np.ndarray of bin_indices
#         intervals: np.ndarray of rows containing (onset_index, offset_index)
#     """
#     onsets = (onsets > onset_threshold).type(torch.int).cpu()
#     frames = (frames > frame_threshold).type(torch.int).cpu()
#     onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

#     pitches = []
#     intervals = []

#     for nonzero in onset_diff.nonzero():
#         frame = nonzero[0].item()
#         pitch = nonzero[1].item()

#         onset = frame
#         offset = frame

#         while onsets[offset, pitch].item() or frames[offset, pitch].item():
#             offset += 1
#             if offset == onsets.shape[0]:
#                 break
#             if (offset != onset) and onsets[offset, pitch].item():
#                 break

#         if offset > onset:
#             pitches.append(pitch)
#             intervals.append([onset, offset])

#     return np.array(pitches), np.array(intervals)


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(
                np.mean(velocity_samples) if len(velocity_samples) > 0 else 0
            )

    return np.array(pitches), np.array(intervals), np.array(velocities)
