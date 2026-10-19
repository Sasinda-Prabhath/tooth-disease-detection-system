import unittest
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.training.evaluate_hybrid import score
from dental_fdi.inference.hybrid_model import HybridFDIModel, ToothRefiner, crop_and_position, refine_rows


class HybridTests(unittest.TestCase):
    def test_refinement_requires_confidence_and_preserves_detector_score(self):
        row = {'pred_fdi': '18', 'conf': .8, 'xyxy': [0, 0, 20, 40]}
        weak = np.full(32, .6 / 31)
        weak[FDI_CODES.index('17')] = .4
        self.assertEqual(refine_rows([row], [weak])[0]['pred_fdi'], '18')
        strong = np.zeros(32)
        strong[FDI_CODES.index('17')] = .95
        strong[FDI_CODES.index('18')] = .05
        refined = refine_rows([row], [strong])[0]
        self.assertEqual(refined['pred_fdi'], '17')
        self.assertEqual(refined['detector_fdi'], '18')
        self.assertLessEqual(refined['conf'], row['conf'])
        self.assertTrue(refined['label_refined'])

    def test_invalid_probabilities_are_rejected(self):
        row = {'pred_fdi': '18', 'conf': .8, 'xyxy': [0, 0, 20, 40]}
        with self.assertRaises(ValueError):
            refine_rows([row], [[float('nan')] * 32])
        with self.assertRaises(ValueError):
            refine_rows([row], [])

    def test_train_and_inference_crops_have_shared_geometry(self):
        crop, position = crop_and_position(Image.new('RGB', (200, 100)), [-4, 10, 50, 90])
        self.assertEqual(tuple(crop.shape), (3, 224, 224))
        self.assertTrue(torch.allclose(position, torch.tensor([0, .1, .25, .9])))
        with self.assertRaises(ValueError):
            crop_and_position(Image.new('RGB', (200, 100)), [50, 10, 40, 90])

    def test_both_encoders_produce_32_logits(self):
        torch.set_num_threads(2)
        for architecture in ('efficientnet_b0', 'resnet50'):
            model = ToothRefiner(architecture).eval()
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 224, 224), torch.zeros(1, 4))
            self.assertEqual(tuple(output.shape), (1, 32))
            self.assertTrue(torch.isfinite(output).all())

    def test_comparison_requires_correct_class_and_overlap(self):
        truth = [{'fdi': '18', 'xyxy': [0, 0, 20, 40]}]
        self.assertEqual(score([dict(truth[0], conf=.9)], truth), (1, 0, 0))
        self.assertEqual(score([dict(truth[0], fdi='17', conf=.9)], truth), (0, 1, 1))

    def test_corrupt_bundle_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.fdi.zip'
            manifest = {'version': 1, 'classes': FDI_CODES,
                        'preprocessing': 'rgb_224_imagenet_pad15_xyxy_v1',
                        'detector_sha256': 'wrong', 'refiner_sha256': 'wrong'}
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('manifest.json', json.dumps(manifest))
                archive.writestr('detector.pt', b'corrupt')
                archive.writestr('refiner.pt', b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt detector'):
                HybridFDIModel(path)


if __name__ == '__main__':
    unittest.main()
