import random

import soundfile as sf
import torch
from torch.utils.data import Dataset


class VocalDataset(Dataset):
    """Generates synthetic (mixture, vocals, accompaniment) triplets from random segments.

    On each call it draws a vocals segment and an accompaniment segment
    (possibly from different tracks), applies random gains and sums them.

    Args:
        tracks: list of Paths, one folder per track, each containing
            vocals.wav and accompaniment.wav.
        duration: length of each segment, in seconds.
        sr: sampling rate of the files, in Hz.
        examples_per_epoch: how many examples count as one epoch.
    """

    def __init__(self, tracks, duration=3.0, sr=44100, examples_per_epoch=2000):
        self.tracks = tracks
        self.n_samples = int(duration * sr)
        self.examples_per_epoch = examples_per_epoch

    def __len__(self):
        return self.examples_per_epoch

    def _random_segment(self, path):
        """Reads a random segment of n_samples samples and returns a 1D tensor."""
        total = sf.info(path).frames
        start = random.randint(0, total - self.n_samples)
        audio, _ = sf.read(path, start=start, stop=start + self.n_samples, dtype="float32")
        return torch.from_numpy(audio)

    def __getitem__(self, idx):
        """Returns (mixture, vocals, accompaniment), each of shape (n_samples,).

        idx is ignored: the sampling is random on every call.
        """
        vocals = self._random_segment(random.choice(self.tracks) / "vocals.wav")
        accompaniment = self._random_segment(random.choice(self.tracks) / "accompaniment.wav")
        vocals = vocals * random.uniform(0.5, 1.2)
        accompaniment = accompaniment * random.uniform(0.5, 1.2)
        return vocals + accompaniment, vocals, accompaniment
