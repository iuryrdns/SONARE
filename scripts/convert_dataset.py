import io
import zipfile
from pathlib import Path

import soundfile as sf

ZIP = Path("musdb18hq.zip")
OUT = Path("data/train")
MONO = True


def read_audio(zf, name):
    """Read an audio file directly from a zip archive and optionally convert to mono.

    Parameters:
        zf (zipfile.ZipFile): Open zip archive object.
        name (str): Path of the file inside the zip archive.

    Returns:
        numpy.ndarray: Audio data array in float32 precision.
    """
    with zf.open(name) as f:
        audio, _ = sf.read(io.BytesIO(f.read()), dtype="float32")
    return audio.mean(axis=1) if MONO else audio


with zipfile.ZipFile(ZIP) as zf:
    """Find all training track folders in the zip, separate vocals from

    accompaniment (drums + bass + other), and export them as WAV files.
    """
    tracks = sorted({
        Path(n).parent.as_posix()
        for n in zf.namelist()
        if "train/" in n and n.endswith("vocals.wav")
    })
    print(f"{len(tracks)} tracks")

    for track in tracks:
        vocals = read_audio(zf, f"{track}/vocals.wav")
        accompaniment = sum(read_audio(zf, f"{track}/{s}.wav") for s in ("drums", "bass", "other"))

        destination = OUT / Path(track).name
        destination.mkdir(parents=True, exist_ok=True)
        sf.write(destination / "vocals.wav", vocals, 44100, subtype="PCM_16")
        sf.write(destination / "accompaniment.wav", accompaniment, 44100, subtype="PCM_16")
        print("ok:", destination.name)