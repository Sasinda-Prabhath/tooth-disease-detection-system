import unittest
from unittest.mock import patch

from PIL import Image

from dental_fdi.analysis.bone_predictions import predict_bone_inventory


class BonePredictionTests(unittest.TestCase):
    def test_every_corrected_tooth_contributes_and_failures_are_unavailable(self):
        image = Image.new('RGB', (300, 300))
        boxes = [{'fdi': fdi, 'xyxy': [10, 10, 100, 250]} for fdi in ('15', '17', '24', '33')]
        valid = [[20, 20], [20, 60], [20, 220], [80, 20], [80, 80], [80, 220]]
        invalid = [[20, 20], [20, 10], [20, 220], [80, 20], [80, 80], [80, 220]]
        with patch('dental_fdi.analysis.bone_predictions.predict_bone_points',
                   side_effect=[(valid, [.9]*6), ([None]*6, [.1]*6),
                                RuntimeError('Inference failed'), (invalid, [.9]*6)]) as predictor:
            results = predict_bone_inventory(object(), image, boxes)
        self.assertEqual(predictor.call_count, 4)
        self.assertEqual(set(results), {'15', '17', '24', '33'})
        self.assertNotIn('error', results['15'])
        self.assertFalse(results['15']['confirmed'])
        for fdi in ('17', '24', '33'):
            self.assertTrue(results[fdi]['error'].startswith('Unavailable:'))
        self.assertNotIn('16', results)


if __name__ == '__main__':
    unittest.main()
