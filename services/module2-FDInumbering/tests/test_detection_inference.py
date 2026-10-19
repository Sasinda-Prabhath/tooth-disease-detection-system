import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from dental_fdi.analysis.dental_analysis import select_detections
from dental_fdi.inference.detection_inference import infer_detections


def result(fdi, confidence):
    box = SimpleNamespace(cls=np.array([0]), conf=np.array([confidence]),
                          xyxy=np.array([[10, 10, 40, 60]]))
    return SimpleNamespace(boxes=[box], names={0: fdi})


class RecoveryTests(unittest.TestCase):
    def test_two_resolutions_can_add_a_missing_class_at_one_percent(self):
        model = Mock()
        model.predict.side_effect = [[result('17', .8)], [result('18', .015)]]
        rows = infer_detections(model, Image.new('RGB', (100, 100)), .01, .5, 1024, True)
        self.assertEqual([b['fdi'] for b in select_detections(rows, 100, 100, .01)], ['17', '18'])
        self.assertEqual([c.kwargs['imgsz'] for c in model.predict.call_args_list], [1024, 1280])
        self.assertTrue(all(c.kwargs['conf'] == .01 for c in model.predict.call_args_list))
        self.assertTrue(all(c.kwargs['agnostic_nms'] is False for c in model.predict.call_args_list))

    def test_one_resolution_is_default_and_missing_teeth_are_not_invented(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=None)]
        self.assertEqual(infer_detections(model, Image.new('RGB', (100, 100)), .01, .5, 1280), [])
        self.assertEqual(model.predict.call_count, 1)


if __name__ == '__main__':
    unittest.main()
