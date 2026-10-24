"""Regression tests for independent calibration and class-sensitive accuracy gates."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper

from export import check_map_drop, quantize_conv_only, resolve_split_paths, stratified_sample


class ExportQualityTests(unittest.TestCase):
    def test_sensitive_layer_exclusion_does_not_change_detect_head_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'fp32.onnx', Path(tmp) / 'int8.onnx'
            weights = [numpy_helper.from_array(np.ones((3, 3, 1, 1), dtype=np.float32), name='weight')]
            graph = helper.make_graph([
                helper.make_node('Conv', ['images', 'weight'], ['features'], name='/model.0/conv/Conv'),
                helper.make_node('Conv', ['features', 'weight'], ['output'], name='/model.23/conv/Conv'),
            ], 'test', [helper.make_tensor_value_info('images', 1, [1, 3, 2, 2])],
                [helper.make_tensor_value_info('output', 1, [1, 3, 2, 2])], weights)
            onnx.save(helper.make_model(graph), source)
            def fake_quantize(**kwargs):
                self.assertEqual(set(kwargs['nodes_to_exclude']), {'/model.0/conv/Conv', '/model.23/conv/Conv'})
                onnx.save(onnx.load(source), kwargs['model_output'])
            with patch('export.quantize_static', side_effect=fake_quantize):
                quantize_conv_only(source, output, [], 2, 'qdq', 'u8', False, fp32_modules=(0,))
            metadata = {item.key: item.value for item in onnx.load(output).metadata_props}
            self.assertEqual(metadata['quantization.excluded_detect_conv_count'], '1')
            self.assertEqual(metadata['quantization.fp32_modules'], '[0]')

    def test_class_regression_cannot_hide_behind_overall_map(self):
        reference = {'map50_95': .70, 'classes': {'bird': {'map50': .76, 'map50_95': .62}}}
        candidate = {'map50_95': .70, 'classes': {'bird': {'map50': .75, 'map50_95': .607}}}
        with self.assertRaisesRegex(RuntimeError, 'bird'):
            check_map_drop(reference, candidate, .003, .005)

    def test_small_measured_changes_pass(self):
        reference = {'map50_95': .70, 'classes': {'bird': {'map50': .76, 'map50_95': .62}}}
        candidate = {'map50_95': .699, 'classes': {'bird': {'map50': .759, 'map50_95': .619}}}
        drops = check_map_drop(reference, candidate, .003, .005)
        self.assertAlmostEqual(drops['classes']['bird']['map50_95'], .001)

    def test_nonfinite_accuracy_does_not_pass_the_gate(self):
        reference = {'map50_95': .70, 'classes': {'bird': {'map50': .76, 'map50_95': .62}}}
        candidate = {'map50_95': float('nan'), 'classes': reference['classes']}
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            check_map_drop(reference, candidate, .003, .005)

    def test_calibration_covers_rare_classes_without_duplicates_or_rng_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir(); (root / 'labels').mkdir()
            paths = []
            for i in range(104):
                path = root / 'images' / f'{i}.jpg'
                path.touch()
                cls = 0 if i < 100 else 1
                (root / 'labels' / f'{i}.txt').write_text(f'{cls} 0.5 0.5 0.5 0.5\n')
                paths.append(path)
            state = random.getstate()
            selected = stratified_sample(paths + paths, 8, 123, 2)
            self.assertEqual(len(selected), 8)
            self.assertEqual(len(set(selected)), 8)
            self.assertEqual(sum(int(p.stem) >= 100 for p in selected), 4)
            self.assertEqual(random.getstate(), state)
            self.assertEqual(selected, stratified_sample(paths, 8, 123, 2))

    def test_split_resolution_deduplicates_multiple_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.jpg').touch()
            (root / 'train.txt').write_text('./a.jpg\n./a.jpg\n')
            (root / 'data.yaml').write_text('path: .\ntrain: [train.txt, train.txt]\nnames: [bird]\n')
            self.assertEqual(resolve_split_paths(root / 'data.yaml', 'train'), [root / 'a.jpg'])


if __name__ == '__main__':
    unittest.main()
