from __future__ import annotations

import numpy as np
import pytest

from ripple.audio.augmentation import add_white_noise, apply_gain
from ripple.audio.framing import StreamingFramer, frame_audio
from ripple.audio.io import AudioBackend, AudioBuffer, load_audio, save_audio
from ripple.audio.pitch import estimate_yin
from ripple.audio.resample import resample_audio, resampled_length


def test_streaming_framing_is_invariant_to_call_grouping() -> None:
    samples = np.arange(1_441, dtype=np.float32)[np.newaxis, :]
    reference = frame_audio(AudioBuffer(samples, 24_000), pad=True)
    framer = StreamingFramer(frame_samples=480, channels=1)
    outputs = [
        framer.push(samples[:, :1]),
        framer.push(samples[:, 1:479]),
        framer.push(samples[:, 479:1000]),
        framer.push(samples[:, 1000:]),
        framer.flush(pad=True),
    ]
    streamed = np.concatenate(outputs)
    np.testing.assert_array_equal(streamed, reference)
    assert streamed.shape == (4, 1, 480)
    assert framer.buffered_samples == 0
    assert not reference.flags.writeable
    assert all(not output.flags.writeable for output in outputs)


def test_resampling_has_exact_length_and_preserves_constant_signal() -> None:
    source = AudioBuffer(np.full((2, 480), 0.25, dtype=np.float32), 24_000)
    downsampled = resample_audio(source, 16_000)
    assert downsampled.frames == resampled_length(480, 24_000, 16_000) == 320
    np.testing.assert_allclose(downsampled.samples, 0.25, atol=1e-6)
    assert not downsampled.samples.flags.writeable

    identity = resample_audio(source, 24_000)
    np.testing.assert_array_equal(identity.samples, source.samples)
    assert identity.samples is not source.samples


def test_yin_fallback_tracks_a_sine_and_rejects_silence() -> None:
    sample_rate = 24_000
    time = np.arange(9_600, dtype=np.float64) / sample_rate
    sine = 0.5 * np.sin(2.0 * np.pi * 200.0 * time)
    estimate = estimate_yin(sine, sample_rate)
    assert estimate.frequency_hz == pytest.approx(200.0, rel=0.01)
    assert estimate.periodicity > 0.95

    silence = estimate_yin(np.zeros_like(sine), sample_rate)
    assert silence.frequency_hz == 0.0
    assert silence.voiced_probability == 0.0


def test_seeded_augmentation_is_repeatable() -> None:
    audio = AudioBuffer(np.full((1, 1_000), 0.1, dtype=np.float32), 24_000)
    first = add_white_noise(audio, 20.0, seed=42)
    second = add_white_noise(audio, 20.0, seed=42)
    np.testing.assert_array_equal(first.audio.samples, second.audio.samples)
    assert not first.time_labels_invalidated
    assert apply_gain(audio, 6.0).audio.samples.max() > audio.samples.max()


def test_soundfile_round_trip(tmp_path) -> None:
    pytest.importorskip("soundfile")
    path = tmp_path / "fixture.wav"
    audio = AudioBuffer(np.linspace(-0.5, 0.5, 480, dtype=np.float32), 24_000)
    save_audio(path, audio, subtype="FLOAT")
    restored = load_audio(path, AudioBackend.SOUNDFILE)
    assert restored.sample_rate == audio.sample_rate
    np.testing.assert_allclose(restored.samples, audio.samples, atol=1e-7)

