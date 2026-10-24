import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format
from ultralytics.nn.tasks import DetectionModel

from train import SCREEN_AUGMENTATION_P, ScratchTrainer, check_training_runtime
from training_augmentations import ScreenCaptureAugmentation
from texture_validation import TextureValidationProbe, localized_class_counts, select_probe_labels


class ScreenCaptureTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(17).integers(0, 256, (96, 128, 3), dtype=np.uint8)

    def test_corruptions_preserve_labels_for_every_class(self):
        for mode in ("moire", "blur", "resample", "jpeg"):
            for class_id in range(8):
                with self.subTest(mode=mode, class_id=class_id):
                    boxes = np.array([[0.5, 0.4, 0.25, 0.2]], dtype=np.float32)
                    expected_boxes = boxes.copy()
                    classes = np.array([[class_id]], dtype=np.float32)
                    labels = {"img": self.image.copy(), "bboxes": boxes, "cls": classes}
                    with patch("training_augmentations.random.choices", return_value=[mode]):
                        output = ScreenCaptureAugmentation(p=1.0)(labels)
                    self.assertIs(output["bboxes"], boxes)
                    self.assertIs(output["cls"], classes)
                    np.testing.assert_array_equal(boxes, expected_boxes)
                    self.assertEqual(output["img"].shape, self.image.shape)
                    self.assertEqual(output["img"].dtype, np.uint8)
                    self.assertFalse(np.array_equal(output["img"], self.image))

    def test_probability_gate_and_worker_seed_reproducibility(self):
        labels = {"img": self.image.copy()}
        original = labels["img"]
        self.assertIs(ScreenCaptureAugmentation(p=0)(labels)["img"], original)
        random.seed(17)
        first = ScreenCaptureAugmentation(p=1)({"img": self.image.copy()})["img"]
        random.seed(17)
        second = ScreenCaptureAugmentation(p=1)({"img": self.image.copy()})["img"]
        np.testing.assert_array_equal(first, second)
        random.seed(17)
        applied = sum(
            not np.array_equal(ScreenCaptureAugmentation()({"img": self.image.copy()})["img"], self.image)
            for _ in range(400)
        )
        self.assertTrue(110 <= applied <= 175, applied)

    def test_strong_screen_patterns_cover_high_frequency_and_color(self):
        # Legacy stripes (amplitude <=18) cannot reach this coverage.
        flat = np.full((256, 256, 3), 180, np.uint8)
        rms, chroma = [], []
        for seed in range(32):
            out = ScreenCaptureAugmentation.apply_image(flat, random.Random(seed), "moire")
            residual = out.astype(float) - cv2.GaussianBlur(out, (0, 0), 1).astype(float)
            rms.append(np.std(residual))
            chroma.append(np.std(out[..., 0].astype(float) - out[..., 2].astype(float)))
        self.assertGreater(max(rms), 32.0)
        self.assertGreater(max(chroma), 20.0)

    def test_probe_rng_does_not_perturb_training_randomness(self):
        state = random.getstate()
        a = ScreenCaptureAugmentation.apply_image(self.image, random.Random(42), "moire")
        b = ScreenCaptureAugmentation.apply_image(self.image, random.Random(42), "moire")
        np.testing.assert_array_equal(a, b)
        self.assertEqual(random.getstate(), state)


class TextureValidationTests(unittest.TestCase):
    def test_incompatible_environment_explains_how_to_activate_yolo(self):
        with patch("train.DEFAULT_CFG_DICT", {"cls": .5}):
            with self.assertRaisesRegex(RuntimeError, "conda activate yolo"):
                check_training_runtime()
        with patch("train.DEFAULT_CFG_DICT", {"cls_pw": .1}):
            check_training_runtime()

    def test_checkpoint_guard_and_score_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = object.__new__(ScratchTrainer)
            trainer.wdir = Path(tmp)
            trainer.last = Path(tmp) / "last.pt"
            trainer.last.write_bytes(b"first checkpoint")
            trainer.metrics = {"probe/score": .5}
            trainer.epoch = 0
            with patch("ultralytics.engine.trainer.BaseTrainer.save_model", return_value=False):
                self.assertFalse(trainer.save_model())
                self.assertFalse((Path(tmp) / "best_texture.pt").exists())
            with patch("ultralytics.engine.trainer.BaseTrainer.save_model", return_value=True):
                trainer.save_model()
                trainer.last.write_bytes(b"worse checkpoint")
                trainer.metrics["probe/score"] = .4
                del trainer._best_texture_score  # Simulate a restarted process.
                trainer.save_model()
                self.assertEqual((Path(tmp) / "best_texture.pt").read_bytes(), b"first checkpoint")

    def test_wrong_class_with_higher_score_is_not_counted_as_correct(self):
        pred = torch.tensor([[0, 0, 10, 10, .8, 2], [0, 0, 10, 10, .6, 1]])
        correct, total = localized_class_counts(pred, torch.tensor([[0, 0, 10, 10]]), torch.tensor([1]), 8)
        self.assertEqual(correct.sum(), 0)
        self.assertEqual(total[1], 1)
        correct, _ = localized_class_counts(pred[1:], torch.tensor([[0, 0, 10, 10]]), torch.tensor([1]), 8)
        self.assertEqual(correct[1], 1)

    def test_selection_is_stable_balanced_and_only_uses_supplied_validation(self):
        labels = [{"im_file": f"val/{i}.jpg", "cls": np.array([[i % 8]])} for i in range(80)]
        a = select_probe_labels(labels, 8, 2)
        b = select_probe_labels(labels[::-1], 8, 2)
        self.assertEqual([x["im_file"] for x in a], [x["im_file"] for x in b])
        self.assertEqual(len(a), 16)
        self.assertTrue(all(x["im_file"].startswith("val/") for x in a))

    def test_probe_is_repeatable_and_restores_model_and_rng(self):
        class FakeModel(torch.nn.Module):
            def forward(self, x):
                return x.new_tensor([[[32, 32, 96, 96, .9, 1]]])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.jpg"
            cv2.imwrite(str(path), np.full((128, 128, 3), 180, np.uint8))
            labels = [{"im_file": str(path), "cls": np.array([[1]]),
                       "bboxes": np.array([[.5, .5, .5, .5]])}]
            probe = TextureValidationProbe(labels, {i: str(i) for i in range(8)}, tmp, sizes=(128,))
            model = FakeModel().train()
            state = random.getstate()
            self.assertEqual(probe(model, "cpu", 1), probe(model, "cpu", 2))
            self.assertTrue(model.training)
            self.assertEqual(random.getstate(), state)


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "images").mkdir()
        (cls.root / "labels").mkdir()
        image = np.random.default_rng(19).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(cls.root / "images/sample.jpg"), image)
        (cls.root / "labels/sample.txt").write_text("1 0.5 0.5 0.4 0.4\n")
        cls.trainer = object.__new__(ScratchTrainer)
        cls.trainer.model = DetectionModel("yolo26n.yaml", nc=8, verbose=False)
        cls.trainer.args = get_cfg(overrides={"imgsz": 128, "workers": 0, "cache": False})
        cls.trainer.data = {"names": {i: str(i) for i in range(8)}, "nc": 8, "channels": 3}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_only_and_tensor_output(self):
        dataset = self.trainer.build_dataset(str(self.root / "images"), mode="train", batch=1)
        stages = dataset.transforms.transforms
        self.assertIsInstance(stages[-2], ScreenCaptureAugmentation)
        self.assertEqual(stages[-2].p, SCREEN_AUGMENTATION_P)
        self.assertIsInstance(stages[-1], Format)
        self.assertEqual(dataset[0]["img"].shape, (3, 128, 128))
        validation = self.trainer.build_dataset(str(self.root / "images"), mode="val", batch=1)
        self.assertFalse(any(isinstance(t, ScreenCaptureAugmentation) for t in validation.transforms.transforms))

    def test_augmentation_survives_closing_mosaic(self):
        dataset = self.trainer.build_dataset(str(self.root / "images"), mode="train", batch=1)
        self.trainer.train_loader = SimpleNamespace(dataset=dataset)
        self.trainer._close_dataloader_mosaic()
        stages = dataset.transforms.transforms
        self.assertEqual(sum(isinstance(t, ScreenCaptureAugmentation) for t in stages), 1)
        self.assertIsInstance(stages[-2], ScreenCaptureAugmentation)
        self.assertIsInstance(stages[-1], Format)
        self.assertEqual(dataset[0]["img"].shape, (3, 128, 128))


if __name__ == "__main__":
    unittest.main()
