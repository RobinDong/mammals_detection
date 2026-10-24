"""Image-only corruptions for screen recapture and resampling robustness."""

import math
import random

import cv2
import numpy as np


class ScreenCaptureAugmentation:
    """Corrupt 35% of images, with labels and geometry unchanged.

    Screen grids account for 5/8 of corruptions; blur, resampling and JPEG for
    1/8 each. Train uses the worker-seeded RNG. apply_image accepts an isolated
    RNG for reproducible validation without changing the training RNG stream.
    """

    def __init__(self, p=0.35):
        if not 0.0 <= p <= 1.0:
            raise ValueError("Screen augmentation probability must be in [0, 1]")
        self.p = p

    def __repr__(self):
        return f"ScreenCaptureAugmentation(v2, p={self.p}, modes=moire:blur:resample:jpeg=5:1:1:1)"

    def __call__(self, labels):
        if random.random() < self.p:
            labels["img"] = self.apply_image(labels["img"], random)
        return labels

    @classmethod
    def apply_image(cls, image, rng, mode=None):
        mode = mode or rng.choices(("moire", "blur", "resample", "jpeg"), weights=(5, 1, 1, 1))[0]
        return getattr(cls, f"_{mode}")(image, rng)

    @staticmethod
    def _moire(image, rng=random):
        """Two crossing grids, RGB phase offsets and spatially varying pitch.

        Periods include frequencies above the sampled image's Nyquist limit;
        sampling them produces lower-frequency aliases. Strength spans mild to
        strong cases. Modulation is brighter on bright surfaces, as in a screen
        photograph. These are synthetic corruptions, not a camera simulator.
        """
        h, w = image.shape[:2]
        x = np.arange(w, dtype=np.float32)[None, :]
        y = np.arange(h, dtype=np.float32)[:, None]
        angle = rng.uniform(0.0, math.pi)
        period = rng.uniform(1.6, 6.0)
        phase = rng.uniform(0.0, 2 * math.pi)
        # A small pitch gradient avoids training on perfectly stationary stripes.
        warp = rng.uniform(-0.10, 0.10) * x * y / max(h, w)
        u = (x * math.cos(angle) + y * math.sin(angle) + warp) * (2 * math.pi / period)
        v = (x * -math.sin(angle) + y * math.cos(angle) - warp) * (
            2 * math.pi / (period * rng.uniform(0.8, 1.2))
        )
        # Retain weak examples; spend half the screen branch on stronger grids.
        strength = rng.uniform(12.0, 32.0) if rng.random() < 0.5 else rng.uniform(32.0, 96.0)
        channel_phase = rng.uniform(0.0, 2.1)
        result = image.astype(np.float32)
        for c in range(3):
            grid = 0.5 * (np.sin(u + phase + c * channel_phase) + np.sin(v + phase - c * channel_phase))
            result[..., c] += strength * (0.35 + 0.65 * result[..., c] / 255.0) * grid
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)

    @staticmethod
    def _blur(image, rng=random):
        # Remove some fine texture while preserving the silhouette and boxes.
        return cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.5))

    @staticmethod
    def _resample(image, rng=random):
        h, w = image.shape[:2]
        scale = rng.uniform(0.5, 0.95)
        small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=rng.choice((cv2.INTER_AREA, cv2.INTER_LINEAR)))
        return cv2.resize(small, (w, h), interpolation=rng.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)))

    @staticmethod
    def _jpeg(image, rng=random):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 95)])
        if not ok:
            raise RuntimeError("Failed to encode training image as JPEG")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
