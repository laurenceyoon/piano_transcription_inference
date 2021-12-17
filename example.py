import os
import argparse
import torch
import time

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio


def inference(audio_path, output_midi_path, device, is_evaluate):
    """Inference template.

    Args:
      model_type: str
      audio_path: str
      cuda: bool
    """

    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path=None)
    """device: 'cuda' | 'cpu'
    checkpoint_path: None for default path, or str for downloaded checkpoint path.
    """

    # Transcribe and write out to MIDI file
    transcribe_time = time.time()
    transcribed_dict = transcriptor.transcribe(audio, output_midi_path)
    print("Transcribe time: {:.3f} s".format(time.time() - transcribe_time))

    if is_evaluate:
        print(f"let's evaluate~~~")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_midi_path", type=str, required=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    args = parser.parse_args()

    audio_path = args.audio_path
    output_midi_path = args.output_midi_path
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    is_evaluate = args.evaluate
    inference(audio_path, output_midi_path, device, is_evaluate)
