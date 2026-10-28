import torch

N_FFT = 2048
HOP_LENGTH = 1024


def compute_stft(waveform: torch.Tensor) -> torch.Tensor:
    """Compute the Short-Time Fourier Transform (STFT) of an input waveform."""
    window = torch.hann_window(N_FFT, device=waveform.device)
    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True,
    )


def magnitude_phase(spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate a complex spectrogram into its magnitude and phase components."""
    return spec.abs(), spec.angle()


def compute_istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    """Convert a complex spectrogram back into a time-domain waveform.

    Args:
        spec: Complex tensor of shape (..., n_freq_bins, n_frames).
        length: Number of samples in the original waveform (for precise padding/cropping).

    Returns:
        Real tensor of shape (..., length).
    """
    window = torch.hann_window(N_FFT, device=spec.device)
    return torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        length=length,
    )
