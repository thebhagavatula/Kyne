import numpy as np

from api.main import _compute_analog_hole_confidence
from kpe.extract.optical_flow import MotionDescriptor


def _make_desc(n: int, valid: bool = True) -> MotionDescriptor:
    return MotionDescriptor(
        vector=np.zeros(10, dtype=np.float32),
        frame_index=0,
        n_points=n,
        valid=valid,
    )


def test_analog_hole_confidence_higher_for_noisy_phone_like_signal():
    clean_descriptors = [_make_desc(180, True) for _ in range(40)]
    noisy_descriptors = (
        [_make_desc(30, True) for _ in range(20)]
        + [_make_desc(0, False) for _ in range(20)]
    )

    clean_sig = np.zeros((40, 10), dtype=np.float32)
    clean_sig[:, -2:] = 0.05

    rng = np.random.default_rng(42)
    noisy_sig = rng.normal(0.0, 1.2, size=(40, 10)).astype(np.float32)

    clean_conf, _ = _compute_analog_hole_confidence(clean_descriptors, clean_sig, 200)
    noisy_conf, _ = _compute_analog_hole_confidence(noisy_descriptors, noisy_sig, 200)

    assert 0.0 <= clean_conf <= 1.0
    assert 0.0 <= noisy_conf <= 1.0
    assert noisy_conf > clean_conf


def test_analog_hole_confidence_zero_when_no_descriptors():
    conf, metrics = _compute_analog_hole_confidence([], np.zeros((1, 10), dtype=np.float32), 200)
    assert conf == 0.0
    assert metrics["invalid_ratio"] == 1.0
