import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.inference.ensemble import ThreeModelFDI, fuse_rows, validate_weights


def probabilities(fdi, value):
    scores = np.full(32, (1 - value) / 31)
    scores[FDI_CODES.index(fdi)] = value
    return scores


class CountingClassifier(torch.nn.Module):
    def __init__(self, fdi):
        super().__init__()
        self.calls = 0
        self.fdi = fdi

    def forward(self, crops, positions):
        self.calls += 1
        return torch.tensor(np.log(probabilities(self.fdi, .8)), dtype=torch.float32).repeat(len(crops), 1)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.row = {'pred_fdi': '18', 'conf': .9, 'xyxy': [10, 10, 40, 90]}

    def test_each_model_changes_the_final_score_even_when_labels_agree(self):
        base, weaker = probabilities('18', .8), probabilities('18', .5)
        result = fuse_rows([self.row], [base], [base])[0]
        for changed in (fuse_rows([self.row], [weaker], [base])[0], fuse_rows([self.row], [base], [weaker])[0]):
            self.assertEqual(changed['pred_fdi'], '18')
            self.assertLess(changed['conf'], result['conf'])
            self.assertNotEqual(changed['fusion_scores'], result['fusion_scores'])
        yolo_changed = fuse_rows([dict(self.row, conf=.7)], [base], [base])[0]
        self.assertNotEqual(yolo_changed['conf'], result['conf'])
        self.assertEqual(set(result['model_contributions']), {'yolo26', 'efficientnet_b0', 'resnet50'})

    def test_joint_classifier_evidence_can_change_fdi(self):
        result = fuse_rows([dict(self.row, conf=.2)], [probabilities('17', .9)], [probabilities('17', .9)])[0]
        self.assertEqual(result['pred_fdi'], '17')
        self.assertEqual(result['detector_fdi'], '18')
        self.assertLessEqual(result['conf'], .2)
        self.assertAlmostEqual(sum(result['fusion_scores']), 1)
        self.assertEqual(result['xyxy'], self.row['xyxy'])

    def test_a_component_cannot_be_disabled(self):
        for weights in [(1, 0, 0), (.5, .5, -1), (float('nan'), 1, 1)]:
            with self.assertRaises(ValueError):
                validate_weights(weights)
        with self.assertRaises(ValueError):
            fuse_rows([self.row], [], [probabilities('18', .8)])

    def test_both_classifiers_execute_for_every_proposal(self):
        model = ThreeModelFDI.__new__(ThreeModelFDI)
        model.device, model.weights, model.detector = torch.device('cpu'), [.5, .25, .25], object()
        model.classifiers = {'efficientnet_b0': CountingClassifier('18'), 'resnet50': CountingClassifier('18')}
        with patch('dental_fdi.inference.detection_inference.infer_detections', return_value=[self.row] * 33):
            outputs = model.predict_rows(Image.new('RGB', (100, 100)), .01, .5, 1024)
        self.assertEqual(len(outputs), 33)
        self.assertTrue(all(classifier.calls == 2 for classifier in model.classifiers.values()))

    def test_empty_detector_output_does_not_invent_teeth(self):
        self.assertEqual(fuse_rows([], [], []), [])

    def test_low_confidence_yolo_evidence_preserves_its_winning_class(self):
        result = fuse_rows([dict(self.row, conf=.01)], [probabilities('18', .8)], [probabilities('18', .8)])[0]
        evidence = result['model_contributions']['yolo26']['evidence']
        self.assertEqual(FDI_CODES[int(np.argmax(evidence))], '18')
        self.assertAlmostEqual(sum(evidence), 1)


if __name__ == '__main__':
    unittest.main()
